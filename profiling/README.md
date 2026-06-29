# Autograd profiling & benchmarking

Tools for measuring the autograd forward/backward pass. Each script includes
**only** `src/autograd.jl` (it is self-contained, depending just on
`LinearAlgebra`), so they run unchanged against any commit.

## Scripts

| File | Purpose |
| --- | --- |
| `ad_profile.jl` | Profile one step: prints time/allocations/node count and writes a flame-graph SVG. |
| `ad_benchmark.jl` | Benchmark forward & backward separately with `BenchmarkTools` statistics. |


Both Julia scripts take optional args: `[src/autograd.jl] [label]`. The label
becomes the SVG suffix and the row/header label in reports.

## Environments

The scripts use throwaway / named Julia environments so they don't touch the
package's own `Project.toml`:

- `@prof` — needs `ProfileSVG` (`Profile` is stdlib):
  ```bash
  julia --project=@prof -e 'using Pkg; Pkg.add("ProfileSVG")'
  ```
- `@bench` — needs `BenchmarkTools`:
  ```bash
  julia --project=@bench -e 'using Pkg; Pkg.add("BenchmarkTools")'
  ```

## Profile the current autograd

```bash
julia --project=@prof profiling/ad_profile.jl
```



Writes `profiling/ad_profile_current.svg` and prints node count, single-step
time, GC time, allocation count and memory. Open the SVG in a browser to see
where time goes (2000 sampled steps).

## Benchmark the current autograd

```bash
julia --project=@bench profiling/ad_benchmark.jl
```

## Profile / benchmark across commits

The scripts only read a commit's `src/autograd.jl`, so check each commit into an
isolated git worktree (keeping your working tree clean) and point the script at
it. The relevant commits:

| commit | label |
| --- | --- |
| `50cfaa2` | `plain` |
| `bb10e98` | `mul` |
| `ea5e92e` | `tape` |


For flame graphs, run the profiler against each commit's worktree. Repeat for
each commit/label pair above:

```bash
git worktree add /tmp/ad-plain 50cfaa2
julia --project=@prof profiling/ad_profile.jl /tmp/ad-plain/src/autograd.jl plain
git worktree remove /tmp/ad-plain
```

This produces one SVG per commit (`profiling/ad_profile_plain.svg`,
`ad_profile_mul.svg`, `ad_profile_tape.svg`). Because the profiler script comes
from your current working tree, `HAS_TAPE` is detected per commit (plain/mul
profile the recursive backward; tape profiles the tape backward).

or run: 

```bash
git worktree add /tmp/ad-plain 50cfaa2
julia --project=@bench profiling/ad_benchmark.jl /tmp/ad-plain/src/autograd.jl plain
git worktree remove /tmp/ad-plain
```
To perform the benchmark on the commit with label 50cfaa2.

## Results

[autograd_profiler_results.md](./results/autograd_profiler_results.md) contains the profiling for the **Scalar vs Vector** based Automatic Differentiation. 

[autograd_benchmark.md](./results/autograd_benchmark.md) contains the benchmark and profiling result for different versions of the autograd. Showing the performane gains accross commits.
