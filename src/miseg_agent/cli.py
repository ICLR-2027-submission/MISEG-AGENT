"""Single-case core inference; saving a preview never commits a mask update."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    route = commands.add_parser("route", help="Select among installed automatic segmentation models.")
    route.add_argument("--target", required=True)
    route.add_argument("--available", nargs="+", required=True)
    route.add_argument("--case-id", required=True)
    route.add_argument("--dataset", default="")
    route.add_argument("--objective", default="accuracy")
    route.add_argument("--model", help="Explicit compatible model choice.")
    analyze = commands.add_parser("analyze", help="Diagnose a supplied mask and propose/review corrective points.")
    analyze.add_argument("--image", required=True)
    analyze.add_argument("--mask", required=True)
    analyze.add_argument("--mask-label", type=int)
    analyze.add_argument("--checkpoint", required=True, help="Factorized ED V2 checkpoint; not a legacy two-channel checkpoint.")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--shape", nargs=3, type=int, default=(256, 256, 256), metavar=("Z", "Y", "X"))
    analyze.add_argument("--device", default="auto")
    analyze.add_argument("--nninteractive-model", help="Model folder with plans.json, dataset.json, and fold_0.")
    analyze.add_argument("--config", help="Optional core settings YAML.")
    analyze.add_argument("--points", help="JSON list of explicitly user-selected points; review and export a joint preview.")
    args = parser.parse_args(argv)
    if args.command == "route":
        from .routing.api import select_initial_model
        result = select_initial_model(args.target, args.available, case_id=args.case_id, dataset_hint=args.dataset,
                                      objective=args.objective, explicit_model=args.model)
        print(json.dumps(result, indent=2))
        return
    from .backends.nninteractive import NNInteractiveTool
    from .diagnosis.runtime import ErrorDiagnosisRuntime
    from .io import load_case
    from .session import MISegAgentSession
    if args.points and not args.nninteractive_model:
        parser.error("--points requires --nninteractive-model for executable review.")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, dict):
        parser.error("Configuration must be a YAML mapping.")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory to preserve earlier review results.")
    case = load_case(args.image, args.mask, shape_zyx=args.shape, mask_label=args.mask_label)
    runtime = ErrorDiagnosisRuntime(args.checkpoint, device=None if args.device == "auto" else args.device)
    backend = NNInteractiveTool(config={"model_dir": args.nninteractive_model, "device": args.device}) if args.nninteractive_model else None
    session = MISegAgentSession(case.image, case.mask, diagnosis=runtime, refinement=backend,
                          spacing_xyz=case.spacing_xyz, config=config)
    probabilities = session.diagnose()
    candidates = session.propose()
    output.mkdir(parents=True, exist_ok=True)
    case.save(output / "workspace_image.nii.gz", case.image, is_mask=False)
    case.save(output / "current_mask.nii.gz", case.mask)
    for name, probability in probabilities.items():
        case.save(output / f"{name}.nii.gz", probability, is_mask=False)
    report = {"mask_version": session.mask_version, "shape_zyx": list(case.image.shape),
              "spacing_xyz": list(case.spacing_xyz), "candidates": candidates, "reviews": [],
              "review_status": "not_requested", "mask_committed": False}
    if backend:
        report["reviews"] = session.review(candidates)
        report["review_status"] = "completed" if all(item["metric_reviewed"] for item in report["reviews"]) else "partial_or_unavailable"
    if args.points:
        selected = json.loads(Path(args.points).read_text(encoding="utf-8"))
        if not isinstance(selected, list) or not selected:
            parser.error("--points must contain a nonempty JSON list of point_xyz/label objects.")
        points = [session.point(item["point_xyz"], item["label"]) for item in selected]
        report["selected_point_reviews"] = session.review(points)
        preview = session.preview(points)
        case.save(output / "preview_workspace.nii.gz", preview.mask)
        case.save(output / "preview_native.nii.gz", preview.mask, native=True)
        report["preview_status"] = "requires_human_inspection"
    (output / "review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved core outputs to {output.resolve()}")


if __name__ == "__main__":
    main()
