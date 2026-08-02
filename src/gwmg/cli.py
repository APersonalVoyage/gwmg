"""gwmg command line.

    gwmg info                 show package and environment status
    gwmg run <config>         run a CosmoSIS pipeline
    gwmg plot <chain>         make corner / alpha contours from chain output
    gwmg validate             self-check of the importable core

`run` sets PIPELINE_DIR to the bundled pipeline (or --pipeline-dir) and requires
COSMOSIS_SRC_DIR and HICLASS_DIR in the environment (see docs/install.md).
Config names without a path resolve against the bundled configs directory.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

from . import __version__, pipeline_dir


def _resolve_config(config, pdir):
    if os.path.exists(config):
        return config
    bundled = os.path.join(pdir, "configs", config)
    if os.path.exists(bundled):
        return bundled
    if not config.endswith(".ini") and os.path.exists(bundled + ".ini"):
        return bundled + ".ini"
    sys.exit("Config not found: %s (looked in cwd and %s/configs)" % (config, pdir))


def cmd_info(args):
    pdir = pipeline_dir()
    print("gwmg %s" % __version__)
    print("pipeline dir : %s" % pdir)
    print("configs      : %s" % ", ".join(sorted(
        f for f in os.listdir(os.path.join(pdir, "configs")) if f.endswith(".ini"))))
    print("\nEnvironment (needed by `gwmg run`):")
    for v in ("COSMOSIS_SRC_DIR", "HICLASS_DIR", "PIPELINE_DIR"):
        print("  %-16s = %s" % (v, os.environ.get(v, "<unset>")))
    print("\ncosmosis on PATH : %s" % (shutil.which("cosmosis") or "no (activate the conda env)"))


def cmd_run(args):
    pdir = os.path.abspath(args.pipeline_dir) if args.pipeline_dir else pipeline_dir()
    env = os.environ.copy()
    env["PIPELINE_DIR"] = pdir
    missing = [v for v in ("COSMOSIS_SRC_DIR", "HICLASS_DIR") if not env.get(v)]
    if missing:
        sys.exit("Missing required env vars: %s. See docs/install.md; run `gwmg info`."
                 % ", ".join(missing))
    if not shutil.which("cosmosis"):
        sys.exit("`cosmosis` not on PATH. Activate the conda env first.")

    config = _resolve_config(args.config, pdir)
    cmd = (["mpirun", "-n", str(args.mpi), "cosmosis", "--mpi", config]
           if args.mpi and args.mpi > 1 else ["cosmosis", config])
    if args.test:
        cmd += ["-p", "runtime.sampler=test"]
    print("[gwmg] PIPELINE_DIR=%s" % pdir)
    print("[gwmg] $ %s" % " ".join(cmd))
    return subprocess.call(cmd, env=env)


def cmd_plot(args):
    from .contours import plot_contours, ChainNotReady
    chains = []
    for spec in args.chains:
        parts = spec.split(":")
        path = parts[0]
        label = parts[1] if len(parts) > 1 else os.path.splitext(os.path.basename(path))[0]
        color = parts[2] if len(parts) > 2 else None
        chains.append((path, label, color))
    try:
        written = plot_contours(chains, outdir=args.outdir,
                                burn_in=args.burn_in, usetex=args.usetex)
    except ChainNotReady as e:
        print("[gwmg] chain not ready: %s. CosmoSIS emcee flushes every `nsteps` "
              "steps; wait for the first 'Done N iterations' line." % e.why)
        return 1
    for f in written:
        print("[gwmg] wrote", f)
    return 0


def cmd_validate(args):
    import numpy as np
    from .likelihood import gw_log_likelihood
    z = np.linspace(1e-4, 2.0, 300)
    dl = (1 + z) * 3e5 / 70.0 * z
    h = 70.0 * np.sqrt(0.3 * (1 + z) ** 3 + 0.7) / 3e5
    ratio = np.ones_like(dl)
    val = gw_log_likelihood(z, dl, h, ratio,
                            d_gw_obs=np.array([40.0, 5300.0]), z_obs=np.array([0.01, 0.82]),
                            sigma_dgw=np.array([11.0, 2500.0]), sigma_z=np.array([1e-4, 1e-4]),
                            v_rms=np.array([500.0, 500.0]))
    ok = np.isfinite(val) and val <= 0.0
    print("gw_log_likelihood self-check: %.4f  ->  %s" % (val, "OK" if ok else "FAIL"))
    return 0 if ok else 1


def _register_emulator(sub):
    """Register the emu-* subcommands if the emulator subpackage is present.

    The emulator is not always shipped; when it is absent these commands simply
    do not appear.
    """
    if importlib.util.find_spec("gwmg.emulator") is None:
        return

    def cmd_emu_gen(args):
        from .emulator import generate_training_set, generate_parallel
        kw = dict(lmax=args.lmax, kmax=args.kmax, zmax=args.zmax, nk=args.nk)
        if args.workers > 1:
            n_ok, n_fail = generate_parallel(args.n, args.outdir, workers=args.workers,
                                             seed=args.seed, **kw)
        else:
            n_ok, n_fail = generate_training_set(args.n, args.outdir, seed=args.seed, **kw)
        print("[gwmg] training set: %d ok, %d hi_class failures -> %s"
              % (n_ok, n_fail, args.outdir))
        return 0

    def cmd_emu_merge(args):
        from .emulator import merge_training_sets
        merge_training_sets(args.dirs, args.outdir)
        return 0

    def cmd_emu_train(args):
        from .emulator import train_emulators
        from .emulator.train import _TRAIN_KW_FAST
        try:
            train_emulators(args.training_dir, args.model_dir,
                            train_kw=_TRAIN_KW_FAST if args.fast else None)
        except ImportError:
            sys.exit("cosmopower not installed. Run: pip install cosmopower")
        return 0

    def cmd_emu_validate(args):
        from .emulator import validate_emulators, write_report
        try:
            report = validate_emulators(args.model_dir, args.test_dir)
        except ImportError:
            sys.exit("cosmopower not installed. Run: pip install cosmopower")
        if args.report:
            print("[gwmg] wrote", write_report(report, args.report))
        return 0

    eg = sub.add_parser("emu-gen", help="generate a hi_class training set (CosmoPower format)")
    eg.add_argument("-n", type=int, required=True, help="number of parameter samples")
    eg.add_argument("--outdir", required=True, help="output directory for the training set")
    eg.add_argument("--lmax", type=int, default=2600)
    eg.add_argument("--kmax", type=float, default=1.0)
    eg.add_argument("--zmax", type=float, default=3.0)
    eg.add_argument("--nk", type=int, default=200)
    eg.add_argument("--seed", type=int, default=0, help="base LHS seed")
    eg.add_argument("--workers", type=int, default=1, help="parallel hi_class processes")
    eg.set_defaults(fn=cmd_emu_gen)

    em = sub.add_parser("emu-merge", help="merge parallel training-set directories")
    em.add_argument("dirs", nargs="+", help="training-set directories to concatenate")
    em.add_argument("--outdir", required=True)
    em.set_defaults(fn=cmd_emu_merge)

    et = sub.add_parser("emu-train", help="train CosmoPower emulators on a training set")
    et.add_argument("training_dir", help="directory from `gwmg emu-gen`")
    et.add_argument("--model-dir", required=True, help="where to save trained emulators")
    et.add_argument("--fast", action="store_true", help="quick 3-stage schedule (first pass)")
    et.set_defaults(fn=cmd_emu_train)

    ev = sub.add_parser("emu-validate", help="check emulator accuracy vs exact hi_class")
    ev.add_argument("test_dir", help="held-out test set (emu-gen with a fresh --seed)")
    ev.add_argument("--model-dir", required=True, help="trained emulators dir")
    ev.add_argument("--report", help="write a text accuracy report to this path")
    ev.set_defaults(fn=cmd_emu_validate)


def main(argv=None):
    p = argparse.ArgumentParser(prog="gwmg", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version="gwmg " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="show package and environment status").set_defaults(fn=cmd_info)

    r = sub.add_parser("run", help="run a CosmoSIS pipeline")
    r.add_argument("config", help="config name (bundled) or path to an .ini")
    r.add_argument("--mpi", type=int, default=1, help="MPI ranks (default 1)")
    r.add_argument("--test", action="store_true", help="single test-sampler evaluation")
    r.add_argument("--pipeline-dir", help="override PIPELINE_DIR (default: bundled)")
    r.set_defaults(fn=cmd_run)

    pl = sub.add_parser("plot", help="make contours from chain output")
    pl.add_argument("chains", nargs="+", help="FILE[:LABEL[:COLOR]] (repeatable)")
    pl.add_argument("--outdir", default=".")
    pl.add_argument("--burn-in", type=float, default=0.3)
    pl.add_argument("--usetex", action="store_true")
    pl.set_defaults(fn=cmd_plot)

    sub.add_parser("validate", help="self-check of the core").set_defaults(fn=cmd_validate)

    _register_emulator(sub)

    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
