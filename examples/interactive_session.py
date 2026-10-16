"""Keep a case in memory while a user selects points and reviews a saved preview."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from miseg_agent.backends.nninteractive import NNInteractiveTool
from miseg_agent.diagnosis.runtime import ErrorDiagnosisRuntime
from miseg_agent.io import load_case
from miseg_agent.session import MISegAgentSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--nninteractive-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory.")
    case = load_case(args.image, args.mask)
    session = MISegAgentSession(
        case.image, case.mask, spacing_xyz=case.spacing_xyz,
        diagnosis=ErrorDiagnosisRuntime(args.checkpoint),
        refinement=NNInteractiveTool(config={"model_dir": args.nninteractive_model}),
    )
    session.diagnose()
    candidates = session.propose()
    reviews = session.review(candidates)
    output.mkdir(parents=True, exist_ok=True)
    case.save(output / "workspace_image.nii.gz", case.image, is_mask=False)
    case.save(output / "current_mask.nii.gz", session.mask)
    (output / "review.json").write_text(json.dumps({"candidates": candidates, "reviews": reviews}, indent=2), encoding="utf-8")
    print(f"Inspect workspace_image.nii.gz, current_mask.nii.gz, and review.json in {output.resolve()}.")
    print("Choose points yourself. Enter a JSON file containing point_xyz/label objects; blank exits.")
    points_path = input("Selected points JSON: ").strip()
    if not points_path:
        return
    items = json.loads(Path(points_path).read_text(encoding="utf-8"))
    selected = [session.point(item["point_xyz"], item["label"]) for item in items]
    selected_reviews = session.review(selected)
    (output / "selected_point_reviews.json").write_text(json.dumps(selected_reviews, indent=2), encoding="utf-8")
    preview = session.preview(selected)
    case.save(output / "preview_native.nii.gz", preview.mask, native=True)
    print(f"Inspect the joint preview: {output / 'preview_native.nii.gz'}")
    if input("Commit this preview after inspection? Type yes: ").strip().lower() == "yes":
        final_mask = session.commit(preview.preview_id, approved=True)
        case.save(output / "accepted_mask.nii.gz", final_mask, native=True)
        print("Saved the user-accepted mask. Run session.diagnose() before requesting the next round.")
    else:
        session.reject(preview.preview_id)
        print("Preview rejected; the current mask is unchanged.")


if __name__ == "__main__":
    main()
