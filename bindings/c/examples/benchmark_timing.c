/*
 * benchmark_timing.c
 *
 * 各 COPP / TOPP 求解器计算耗时对比。
 *
 * 计时范围：problem 描述符构建 + 求解器调用。
 * 不计时  ：路径求值、robot 约束摄入、amax_substitute
 *           （所有方法共用的预处理步骤，不影响横向对比）。
 *
 * 方法列表
 * --------
 *   TOPP2-RA            后向可达集 DP + 前向贪心，O(n*m)
 *   COPP2-SOCP          二阶凸目标（时间），Clarabel SOCP
 *   TOPP3-LP  iter 1/2  三阶 LP，第 1/2 次 SCP 迭代
 *   TOPP3-SOCP iter 1/2 三阶 SOCP，第 1/2 次 SCP 迭代
 *   COPP3-SOCP iter 1/2 三阶凸目标（时间 + 热能），第 1/2 次 SCP 迭代
 *
 * 编译（加入 CMakeLists.txt）：
 *   add_copp_c_example(benchmark_timing examples/benchmark_timing.c)
 * 运行：
 *   ./benchmark_timing
 */

/* ── 高精度计时器（平台适配）────────────────────────────────────────── */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static double bench_now_ms(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <time.h>
static double bench_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}
#endif

#include <float.h>
#include <math.h>
#include <stdio.h>

#include "example_common.h"

/* ── 基准记录表 ──────────────────────────────────────────────────────── */

#define MAX_BENCH 16

typedef struct
{
    const char *label;
    double      solve_ms;
    double      t_final;
    int         is_summary; /* 1 = 合计行（两次迭代之和），用 -> 标注 */
} BenchEntry;

static BenchEntry g_bench[MAX_BENCH];
static int        g_bench_n = 0;

static void bench_push(const char *label, double ms, double tf, int summary)
{
    if (g_bench_n >= MAX_BENCH)
        return;
    g_bench[g_bench_n].label      = label;
    g_bench[g_bench_n].solve_ms   = ms;
    g_bench[g_bench_n].t_final    = tf;
    g_bench[g_bench_n].is_summary = summary;
    ++g_bench_n;
}

/* ── 辅助：从二阶 profile 提取 t_final（不计时）──────────────────── */
static int get_tf_2nd(const double *s, size_t n,
                      struct CoppVecF64 a, double *out_tf)
{
    struct CoppVecF64 t_s = {NULL, 0, 0};
    enum CoppStatus status = copp_s_to_t_2nd(
        (struct CoppSliceF64){s, n},
        (struct CoppSliceF64){a.data, a.len},
        0.0, out_tf, &t_s);
    copp_vec_f64_free(t_s);
    return example_expect_ok(status, "copp_s_to_t_2nd");
}

/* ── 辅助：从三阶 profile 提取 t_final（不计时）──────────────────── */
static int get_tf_3rd(const double *s, size_t n,
                      struct CoppProfile3rd prof, double *out_tf)
{
    struct CoppVecF64 t_s = {NULL, 0, 0};
    enum CoppStatus status = copp_s_to_t_3rd(
        (struct CoppSliceF64){s, n},
        (struct CoppSliceF64){prof.a.data, prof.a.len},
        (struct CoppSliceF64){prof.b.data, prof.b.len},
        prof.num_stationary_start, prof.num_stationary_end,
        0.0, out_tf, &t_s);
    copp_vec_f64_free(t_s);
    return example_expect_ok(status, "copp_s_to_t_3rd");
}

/* ── 主程序 ───────────────────────────────────────────────────────────── */

int main(void)
{
    const size_t N   = EXAMPLE_NUM_POINTS; /* 1001 */
    const int    DIM = EXAMPLE_DIM;        /* 3    */

    double s[EXAMPLE_NUM_POINTS];

    /* 路径句柄 */
    struct CoppPath *path2 = NULL; /* 二阶路径（TOPP2 / COPP2 用） */
    struct CoppPath *path3 = NULL; /* 三阶路径（TOPP3 / COPP3 用） */

    /* robot 句柄 */
    struct CoppRobot *robot2 = NULL; /* 仅含 vel / acc 约束 */
    struct CoppRobot *robot3 = NULL; /* 含 vel / acc / jerk 约束 */

    /* 二阶输出 */
    struct CoppVecF64 a_ra    = {NULL, 0, 0};
    struct CoppVecF64 a_socp2 = {NULL, 0, 0};

    /* 三阶种子 */
    struct CoppVecF64 a_seed = {NULL, 0, 0};

    /* 三阶输出 */
    struct CoppProfile3rd prof_lp1   = {{NULL, 0, 0}, {NULL, 0, 0}, 0, 0};
    struct CoppProfile3rd prof_lp2   = {{NULL, 0, 0}, {NULL, 0, 0}, 0, 0};
    struct CoppProfile3rd prof_socp1 = {{NULL, 0, 0}, {NULL, 0, 0}, 0, 0};
    struct CoppProfile3rd prof_socp2 = {{NULL, 0, 0}, {NULL, 0, 0}, 0, 0};
    struct CoppProfile3rd prof_copp1 = {{NULL, 0, 0}, {NULL, 0, 0}, 0, 0};
    struct CoppProfile3rd prof_copp2 = {{NULL, 0, 0}, {NULL, 0, 0}, 0, 0};

    double          t0, t1, ms, tf;
    double          tf_ra = 1.0; /* TOPP2-RA 基准时长，用于 t/t_ra 比值 */
    enum CoppStatus status;
    int             rc = 1;

    /* ── 公共预处理（不计时）────────────────────────────────────────── */
    printf("建立路径与约束（共用预处理，不计时）…\n");

    example_fill_stations(s, N);

    if (example_create_analytic_path_2nd(&path2))
        goto cleanup;
    if (example_create_analytic_path_3rd(&path3))
        goto cleanup;
    if (example_create_robot_2nd(path2, s, N, &robot2))
        goto cleanup;
    if (example_create_robot_3rd(path3, s, N, &robot3))
        goto cleanup;

    printf("开始各求解器计时…\n\n");

    /* ─── 1. TOPP2-RA ──────────────────────────────────────────────── */
    {
        struct Topp2RaOptions opts;
        status = topp2_ra_default_options(&opts);
        if (example_expect_ok(status, "topp2_ra_default_options"))
            goto cleanup;

        struct Topp2Problem prob = {robot2, 0, N - 1, 0.0, 0.0};

        t0     = bench_now_ms();
        status = topp2_ra(prob, opts, &a_ra);
        t1     = bench_now_ms();

        if (example_expect_ok(status, "topp2_ra"))
            goto cleanup;
        if (get_tf_2nd(s, N, a_ra, &tf))
            goto cleanup;

        ms    = t1 - t0;
        tf_ra = tf;
        printf("  TOPP2-RA           %8.2f ms  tf=%.4f s\n", ms, tf);
        bench_push("TOPP2-RA", ms, tf, 0);
    }

    /* ─── 2. COPP2-SOCP（时间目标）────────────────────────────────── */
    {
        struct CoppClarabelOptions opts;
        status = copp_clarabel_default_options(&opts);
        if (example_expect_ok(status, "copp_clarabel_default_options"))
            goto cleanup;
        opts.allow_almost_solved          = true;
        opts.allow_max_iterations         = true;
        opts.allow_insufficient_progress  = true;

        struct CoppSliceF64 empty = {NULL, 0};
        struct CoppObjective objectives[1] = {
            {COPP_OBJECTIVE_KIND_TIME, 1.0, empty, empty, empty},
        };
        struct Copp2Problem prob = {robot2, 0, N - 1, 0.0, 0.0,
                                    objectives, 1};

        t0     = bench_now_ms();
        status = copp2_socp(prob, opts, &a_socp2);
        t1     = bench_now_ms();

        if (example_expect_ok(status, "copp2_socp"))
            goto cleanup;
        if (get_tf_2nd(s, N, a_socp2, &tf))
            goto cleanup;

        ms = t1 - t0;
        printf("  COPP2-SOCP (Time)  %8.2f ms  tf=%.4f s\n", ms, tf);
        bench_push("COPP2-SOCP (Time)", ms, tf, 0);
    }

    /* ── 三阶公用预处理（不计时）──────────────────────────────────── */
    /* 用 robot3 运行 TOPP2-RA 获得线性化种子，并收紧 amax */
    if (example_seed_third_order_problem(robot3, N, &a_seed))
        goto cleanup;

    /* 三阶方法共用的 Clarabel 选项 */
    struct CoppClarabelOptions opts3;
    status = copp_clarabel_default_options(&opts3);
    if (example_expect_ok(status, "copp_clarabel_default_options [3rd]"))
        goto cleanup;
    opts3.allow_almost_solved         = true;
    opts3.allow_max_iterations        = true;
    opts3.allow_insufficient_progress = true;

    /* ─── 3. TOPP3-LP  iter 1 ────────────────────────────────────── */
    {
        struct Topp3Problem prob = {
            robot3, 0,
            (struct CoppSliceF64){a_seed.data, a_seed.len},
            0.0, 0.0, 0.0, 0.0, 1, 1, 1e-10
        };

        t0     = bench_now_ms();
        status = topp3_lp(prob, opts3, &prof_lp1);
        t1     = bench_now_ms();

        if (example_expect_ok(status, "topp3_lp iter 1"))
            goto cleanup;
        if (get_tf_3rd(s, N, prof_lp1, &tf))
            goto cleanup;

        ms = t1 - t0;
        printf("  TOPP3-LP   iter 1  %8.2f ms  tf=%.4f s\n", ms, tf);
        bench_push("TOPP3-LP  iter 1", ms, tf, 0);
    }

    /* ─── 4. TOPP3-LP  iter 2 ────────────────────────────────────── */
    {
        struct Topp3Problem prob = {
            robot3, 0,
            (struct CoppSliceF64){prof_lp1.a.data, prof_lp1.a.len},
            0.0, 0.0, 0.0, 0.0, 1, 1, 1e-10
        };

        t0     = bench_now_ms();
        status = topp3_lp(prob, opts3, &prof_lp2);
        t1     = bench_now_ms();

        if (example_expect_ok(status, "topp3_lp iter 2"))
            goto cleanup;
        if (get_tf_3rd(s, N, prof_lp2, &tf))
            goto cleanup;

        ms = t1 - t0;
        printf("  TOPP3-LP   iter 2  %8.2f ms  tf=%.4f s\n", ms, tf);
        bench_push("TOPP3-LP  iter 2", ms, tf, 0);
        /* 合计行：iter1.solve_ms 在 g_bench[g_bench_n-2] */
        bench_push("TOPP3-LP  total",
                   g_bench[g_bench_n - 2].solve_ms + ms, tf, 1);
    }

    /* ─── 5. TOPP3-SOCP iter 1 ──────────────────────────────────── */
    {
        struct Topp3Problem prob = {
            robot3, 0,
            (struct CoppSliceF64){a_seed.data, a_seed.len},
            0.0, 0.0, 0.0, 0.0, 1, 1, 1e-10
        };

        t0     = bench_now_ms();
        status = topp3_socp(prob, opts3, &prof_socp1);
        t1     = bench_now_ms();

        if (example_expect_ok(status, "topp3_socp iter 1"))
            goto cleanup;
        if (get_tf_3rd(s, N, prof_socp1, &tf))
            goto cleanup;

        ms = t1 - t0;
        printf("  TOPP3-SOCP iter 1  %8.2f ms  tf=%.4f s\n", ms, tf);
        bench_push("TOPP3-SOCP iter 1", ms, tf, 0);
    }

    /* ─── 6. TOPP3-SOCP iter 2 ──────────────────────────────────── */
    {
        struct Topp3Problem prob = {
            robot3, 0,
            (struct CoppSliceF64){prof_socp1.a.data, prof_socp1.a.len},
            0.0, 0.0, 0.0, 0.0, 1, 1, 1e-10
        };

        t0     = bench_now_ms();
        status = topp3_socp(prob, opts3, &prof_socp2);
        t1     = bench_now_ms();

        if (example_expect_ok(status, "topp3_socp iter 2"))
            goto cleanup;
        if (get_tf_3rd(s, N, prof_socp2, &tf))
            goto cleanup;

        ms = t1 - t0;
        printf("  TOPP3-SOCP iter 2  %8.2f ms  tf=%.4f s\n", ms, tf);
        bench_push("TOPP3-SOCP iter 2", ms, tf, 0);
        bench_push("TOPP3-SOCP total",
                   g_bench[g_bench_n - 2].solve_ms + ms, tf, 1);
    }

    /* ─── 7. COPP3-SOCP iter 1（时间 + 热能目标）──────────────── */
    {
        double normalize[EXAMPLE_DIM] = {1.0, 1.0, 1.0};
        struct CoppSliceF64 empty      = {NULL, 0};
        struct CoppSliceF64 norm_slice = {normalize, EXAMPLE_DIM};
        struct CoppObjective objectives[2] = {
            {COPP_OBJECTIVE_KIND_TIME,          1.0, empty, empty, empty},
            {COPP_OBJECTIVE_KIND_THERMAL_ENERGY, 0.1, empty, empty, norm_slice},
        };
        struct Copp3Problem prob = {
            robot3, 0,
            (struct CoppSliceF64){a_seed.data, a_seed.len},
            0.0, 0.0, 0.0, 0.0, 1, 1, 1e-10,
            objectives, 2
        };

        t0     = bench_now_ms();
        status = copp3_socp(prob, opts3, &prof_copp1);
        t1     = bench_now_ms();

        if (example_expect_ok(status, "copp3_socp iter 1"))
            goto cleanup;
        if (get_tf_3rd(s, N, prof_copp1, &tf))
            goto cleanup;

        ms = t1 - t0;
        printf("  COPP3-SOCP iter 1  %8.2f ms  tf=%.4f s\n", ms, tf);
        bench_push("COPP3-SOCP iter 1", ms, tf, 0);
    }

    /* ─── 8. COPP3-SOCP iter 2 ──────────────────────────────────── */
    {
        double normalize[EXAMPLE_DIM] = {1.0, 1.0, 1.0};
        struct CoppSliceF64 empty      = {NULL, 0};
        struct CoppSliceF64 norm_slice = {normalize, EXAMPLE_DIM};
        struct CoppObjective objectives[2] = {
            {COPP_OBJECTIVE_KIND_TIME,          1.0, empty, empty, empty},
            {COPP_OBJECTIVE_KIND_THERMAL_ENERGY, 0.1, empty, empty, norm_slice},
        };
        struct Copp3Problem prob = {
            robot3, 0,
            (struct CoppSliceF64){prof_copp1.a.data, prof_copp1.a.len},
            0.0, 0.0, 0.0, 0.0, 1, 1, 1e-10,
            objectives, 2
        };

        t0     = bench_now_ms();
        status = copp3_socp(prob, opts3, &prof_copp2);
        t1     = bench_now_ms();

        if (example_expect_ok(status, "copp3_socp iter 2"))
            goto cleanup;
        if (get_tf_3rd(s, N, prof_copp2, &tf))
            goto cleanup;

        ms = t1 - t0;
        printf("  COPP3-SOCP iter 2  %8.2f ms  tf=%.4f s\n", ms, tf);
        bench_push("COPP3-SOCP iter 2", ms, tf, 0);
        bench_push("COPP3-SOCP total",
                   g_bench[g_bench_n - 2].solve_ms + ms, tf, 1);
    }

    /* ── 汇总表格 ─────────────────────────────────────────────────── */
    printf("\n");
    printf("+---------------------+----------+------------+---------+\n");
    printf("|  COPP 求解器耗时对比   n=%-4zu  dim=%d  Lissajous       |\n",
           N, DIM);
    printf("+---------------------+----------+------------+---------+\n");
    printf("|  %-19s  %8s  %10s  %7s  |\n",
           "求解器", "耗时(ms)", "t_final(s)", "t/t_ra");
    printf("+---------------------+----------+------------+---------+\n");
    for (int i = 0; i < g_bench_n; ++i)
    {
        const char *mark = g_bench[i].is_summary ? "->" : "  ";
        printf("|%s %-19s  %8.2f  %10.4f  %7.4f  |\n",
               mark,
               g_bench[i].label,
               g_bench[i].solve_ms,
               g_bench[i].t_final,
               g_bench[i].t_final / tf_ra);
    }
    printf("+---------------------+----------+------------+---------+\n");
    printf("  -> 行表示两次 SCP 迭代的合计耗时\n");

    rc = 0;

cleanup:
    copp_profile_3rd_free(prof_copp2);
    copp_profile_3rd_free(prof_copp1);
    copp_profile_3rd_free(prof_socp2);
    copp_profile_3rd_free(prof_socp1);
    copp_profile_3rd_free(prof_lp2);
    copp_profile_3rd_free(prof_lp1);
    copp_vec_f64_free(a_seed);
    copp_vec_f64_free(a_socp2);
    copp_vec_f64_free(a_ra);
    copp_robot_free(robot3);
    copp_robot_free(robot2);
    copp_path_free(path3);
    copp_path_free(path2);
    return rc;
}
