import argparse
import json
import pickle
import shutil
from pathlib import Path
from subprocess import run

import numpy as np
import tqdm.auto as tqdm

from utils import (
    compute_betas,
    compute_rps,
    compute_sign_class,
    compute_valid_frames,
    copy_frames,
    create_video,
    extract_frames,
    get_end,
    run_mediapipe_hands,
    run_vitpose,
    segment_signs,
)


def compute_smpl_x_poses(*, rps_folder, hand, result_folder, images_folder, valid_frames, weights):
    rps_folder = rps_folder.joinpath("data")
    rps_images_folder = rps_folder.joinpath("images")
    shutil.rmtree(rps_images_folder, ignore_errors=True)
    rps_images_folder.mkdir(parents=True)
    rps_keypoints_folder = rps_folder.joinpath("keypoints")
    shutil.rmtree(rps_keypoints_folder, ignore_errors=True)
    rps_keypoints_folder.mkdir(parents=True)

    # Do it inside a loop due to ncg memory issue:
    for frame_int in tqdm.tqdm(valid_frames):
        frame_prefix = f"{frame_int:03}"
        rps_image_path = rps_images_folder.joinpath(f"{frame_prefix}.png")
        rps_image_path.symlink_to(images_folder.joinpath(f"{frame_prefix}.png"))
        rps_keypoint_path = rps_keypoints_folder.joinpath(f"{frame_prefix}_keypoints.json")

        confidence = weights[frame_int] + 0.5
        mp_keypoints_path = result_folder.joinpath(
            "mp_keypoints_{:.1f}".format(confidence), f"{frame_prefix}_keypoints.json"
        )
        rps_keypoint_path.unlink(missing_ok=True)
        rps_keypoint_path.symlink_to(mp_keypoints_path)

        call_smplify_x(data_folder=rps_folder, output_folder=result_folder.joinpath("rps", hand))

        rps_image_path.unlink()
        rps_keypoint_path.unlink()

    return valid_frames


def call_smplify_x(*, data_folder, output_folder):
    return run(
        [
            "python",
            "smplifyx/main.py",
            "--config",
            "cfg_files/fit_smplifyx_ref_sv.yaml",
            "--data_folder",
            data_folder,
            "--output_folder",
            output_folder,
        ],
        check=True,
    )


def call_sgnify_0(
    *,
    output_folder,
    data_folder,
    beta_path,
    expression_path,
    use_symmetry,
    symmetry_weight,
    left_handpose_path,
    right_handpose_path,
    left_reference_weight,
    right_reference_weight,
):
    return run(
        [
            "python",
            "smplifyx/main.py",
            "--config",
            "cfg_files/fit_sgnifyx_sv.yaml",
            "--output_folder",
            output_folder,
            "--data_folder",
            data_folder,
            "--start_opt_stage",
            "0",
            "--loss_type",
            "sgnify",
            "--beta_precomputed",
            "True",
            "--beta_path",
            beta_path,
            "--expression_precomputed",
            "True",
            "--expression_path",
            expression_path,
            "--use_symmetry",
            str(use_symmetry),
            "--symmetry_weight",
            str(symmetry_weight),
            "--left_handpose_path",
            left_handpose_path,
            "--left_reference_weight",
            str(left_reference_weight),
            "--right_handpose_path",
            right_handpose_path,
            "--right_reference_weight",
            str(right_reference_weight),
        ],
        check=True,
    )


def call_sgnify(
    *,
    output_folder,
    data_folder,
    prev_res_path,
    expression_path,
    use_symmetry,
    symmetry_weight,
    left_handpose_path,
    right_handpose_path,
    left_reference_weight,
    right_reference_weight,
):
    return run(
        [
            "python",
            "smplifyx/main.py",
            "--config",
            "cfg_files/fit_sgnifyx_sv.yaml",
            "--output_folder",
            output_folder,
            "--data_folder",
            data_folder,
            "--prev_res_path",
            prev_res_path,
            "--expression_precomputed",
            "True",
            "--expression_path",
            expression_path,
            "--use_symmetry",
            str(use_symmetry),
            "--symmetry_weight",
            str(symmetry_weight),
            "--left_handpose_path",
            left_handpose_path,
            "--left_reference_weight",
            str(left_reference_weight),
            "--right_handpose_path",
            right_handpose_path,
            "--right_reference_weight",
            str(right_reference_weight),
            "--beta_precomputed",
            "True",
        ],
        check=True,
    )


def call_spectre(*, images_folder, output_folder):
    run(
        ["python", "utils/spectre_demo.py", "--images_folder", images_folder, "--output_folder", output_folder],
        check=True,
    )


def run_sgnify(
    result_folder,
    output_folder,
    spectre_folder,
    reference,
    use_symmetry,
    symmetry,
    beta_path,
    left_interp_folder,
    right_interp_folder,
    segment_path,
):
    for frame_int in tqdm.trange(1, get_end(result_folder) + 1):
        frame_prefix = f"{frame_int:03}"
        prev_res_path = output_folder.joinpath("results", f"{frame_int-1:03}.pkl")

        tmp_data_path = result_folder.joinpath("tmp", "data")
        shutil.rmtree(tmp_data_path, ignore_errors=True)
        tmp_data_path.mkdir(parents=True)

        path = tmp_data_path.joinpath("images")
        path.mkdir()
        path.joinpath(f"{frame_prefix}.png").symlink_to(result_folder.joinpath("images", f"{frame_prefix}.png"))

        path = tmp_data_path.joinpath("keypoints")
        path.mkdir()
        path.joinpath(f"{frame_prefix}_keypoints.json").symlink_to(
            result_folder.joinpath("keypoints", f"{frame_prefix}_keypoints.json")
        )

        right_handpose_path = "None"
        left_handpose_path = "None"

        rps_folder = result_folder.joinpath("rps")
        if args.sign_class[1] == "a":
            right_handpose_path = rps_folder.joinpath("ref_right.pkl")
            if args.sign_class[0] != "0":
                left_handpose_path = rps_folder.joinpath("ref_left.pkl")
        else:
            right_handpose_path = right_interp_folder.joinpath(f"{(frame_int-1):03}.pkl")
            if args.sign_class[0] == "1":
                left_handpose_path = left_interp_folder.joinpath(f"{(frame_int-1):03}.pkl")

        expression_path = spectre_folder.joinpath(f"spectre_{frame_int}.pkl")

        left_reference_weight = reference
        right_reference_weight = reference
        symmetry_weight = symmetry

        with segment_path.open() as json_file:
            segment = json.load(json_file)

        if frame_int < segment["t1"] or frame_int > segment["t4"]:
            left_reference_weight = left_reference_weight / 5
            right_reference_weight = right_reference_weight / 5
            symmetry_weight = symmetry_weight / 5

        if frame_int == 1:
            call_sgnify_0(
                output_folder=output_folder,
                data_folder=tmp_data_path,
                beta_path=beta_path,
                expression_path=expression_path,
                left_handpose_path=left_handpose_path,
                right_handpose_path=right_handpose_path,
                left_reference_weight=left_reference_weight,
                right_reference_weight=right_reference_weight,
                use_symmetry=use_symmetry,
                symmetry_weight=symmetry_weight,
            )
        else:
            call_sgnify(
                output_folder=output_folder,
                data_folder=tmp_data_path,
                prev_res_path=prev_res_path,
                expression_path=expression_path,
                use_symmetry=use_symmetry,
                symmetry_weight=symmetry_weight,
                left_handpose_path=left_handpose_path,
                right_handpose_path=right_handpose_path,
                left_reference_weight=left_reference_weight,
                right_reference_weight=right_reference_weight,
            )


def main(args):
    if args.video_path != "None":
        video_path = Path(args.video_path).resolve()
        output_folder = Path(args.output_folder).joinpath(video_path.stem).resolve()
    elif args.image_dir_path != "None":
        image_dir_path = Path(args.image_dir_path).resolve()
        output_folder = Path(args.output_folder).joinpath(image_dir_path.stem).resolve()

    output_folder.mkdir(exist_ok=True, parents=True)

    result_folder = output_folder.joinpath(".tmp")
    result_folder.mkdir(exist_ok=True, parents=True)

    segment_path = result_folder.joinpath("segmentation.json")

    images_folder = result_folder.joinpath("images")
    images_folder.mkdir(exist_ok=True, parents=True)

    openpose_folder = result_folder.joinpath("keypoints_OP")
    openpose_folder.mkdir(exist_ok=True, parents=True)

    mediapipe_folder = result_folder.joinpath("keypoints")

    rps_folder = result_folder.joinpath("rps")
    rps_folder.mkdir(exist_ok=True, parents=True)

    right_interp_folder = rps_folder.joinpath("interp_right")
    right_interp_folder.mkdir(exist_ok=True)

    left_interp_folder = rps_folder.joinpath("interp_left")
    left_interp_folder.mkdir(exist_ok=True)

    beta_path = result_folder.joinpath("median_betas.pkl")

    spectre_folder = result_folder.joinpath("spectre")
    spectre_folder.mkdir(exist_ok=True, parents=True)

    sign_class_path = result_folder.joinpath("sign_class.txt")

    # Invariance constraint
    reference = 90
    symmetry = 90

    # PRE-PROCESS:
    # 0. Extract frames from video
    # 1. Run vitpose and mediapipe on all frames
    # 2. Segment video
    # 3. Run SMPLify-X on frames selected
    # 4. Find RPS
    # 5. Find shape betas
    # 6. Run SPECTRE
    if args.skip_preprocessing:
        print("Skipping preprocessing...")
        assert args.sign_class in ["0a", "0b", "1a", "1b", "2a", "2b", "3a", "3b", "-1"]

        if args.sign_class == "-1":
            args.sign_class = sign_class_path.read_text().strip()
    else:
        if args.video_path != "None":
            # 0. Extract frames from video
            print("Extracting frames with FFmpeg...")
            extract_frames(video_path=video_path, output_folder=result_folder.joinpath("images"))
        else:
            # 0. Copy frames from folder
            print("Copying frames...")
            copy_frames(image_dir_path=image_dir_path, output_folder=result_folder.joinpath("images"))

        # 1. Run VitPose and MediaPipe on each frame
        # VitPose
        print("Extracting 2D keypoints with ViTPose...")
        run_vitpose(images_folder=images_folder, output_folder=openpose_folder)

        # MediaPipe
        print("Extracting 2D keypoints with MediaPipe for RPS...")
        confidences = np.arange(0.6, 1.0, 0.1)
        for confidence in confidences:
            mp_folder = result_folder.joinpath("mp_keypoints_{:.1f}".format(confidence))
            shutil.rmtree(mp_folder, ignore_errors=True)
            shutil.copytree(openpose_folder, mp_folder)
            run_mediapipe_hands(
                output_folder=result_folder, confidence=confidence, static_image_mode=True, keypoint_folder=mp_folder
            )

        mp_files = ["mp_keypoints_0.6.pkl", "mp_keypoints_0.7.pkl", "mp_keypoints_0.8.pkl", "mp_keypoints_0.9.pkl"]
        mp_results = []
        for pkl_file in mp_files:
            with (result_folder.joinpath(pkl_file)).open("rb") as file:
                mp_results.append(pickle.load(file))

        valid_frames_right = (np.ones((get_end(result_folder) + 1))) * -1.0
        valid_frames_left = (np.ones((get_end(result_folder) + 1))) * -1.0

        for weight in range(len(mp_files)):
            valid_frames_right[list(map(int, mp_results[weight]["Right"]))] = (weight + 1) / 10
            valid_frames_left[list(map(int, mp_results[weight]["Left"]))] = (weight + 1) / 10

        output = {"Right": valid_frames_right, "Left": valid_frames_left}

        weights_path = result_folder.joinpath("mp_keypoints_weight.pkl")
        with weights_path.open("wb") as json_file:
            pickle.dump(output, json_file)

        # 2. Segment video
        print("Segmenting signs...")
        segment_signs(openpose_folder=openpose_folder, output_path=segment_path)

        with segment_path.open() as json_file:
            segment = json.load(json_file)
        reconstruct_left, reconstruct_right = compute_valid_frames(result_folder, segment)

        # 3. Run SMPLify-X on frames selected that have MP keypoints
        # This provides the RPS and betas
        # Link images and keypoints of the selected frames in a new folder
        if np.any(valid_frames_right > -1):
            print("Running SMPLify-X inside segmentation window for right hand...")
            compute_smpl_x_poses(
                rps_folder=rps_folder,
                result_folder=result_folder,
                images_folder=images_folder,
                valid_frames=reconstruct_right,
                weights=valid_frames_right,
                hand="right",
            )

        if np.any(valid_frames_left > -1):
            print("Running SMPLify-X inside segmentation window for left hand...")
            compute_smpl_x_poses(
                rps_folder=rps_folder,
                result_folder=result_folder,
                images_folder=images_folder,
                valid_frames=reconstruct_left,
                weights=valid_frames_left,
                hand="left",
            )

        print("Finding sign class...")
        compute_sign_class(
            openpose_folder=openpose_folder,
            result_folder=result_folder,
            segment_path=segment_path,
            sign_class_path=sign_class_path,
        )

        if args.sign_class == "-1":
            args.sign_class = sign_class_path.read_text().strip()

        # 4. Find RPS (for now we do not have a weighted average)
        # for now we only have the rps for subclasses A
        print("Finding the RPS using valid MediaPipe frames...")
        compute_rps(
            sign_class=args.sign_class,
            rps_folder=rps_folder,
            result_folder=result_folder,
            right_interp_folder=right_interp_folder,
            left_interp_folder=left_interp_folder,
            segment_path=segment_path,
        )

        # 5. Find betas
        print("Finding betas...")
        compute_betas(rps_folder=rps_folder, beta_path=beta_path)

        # 6. Run SPECTRE
        print("Running SPECTRE...")
        call_spectre(images_folder=images_folder, output_folder=spectre_folder)

        # MediaPipe
        print("Extracting 2D keypoints with MediaPipe for SGNify...")
        shutil.rmtree(mediapipe_folder, ignore_errors=True)
        shutil.copytree(openpose_folder, mediapipe_folder)
        run_mediapipe_hands(
            output_folder=result_folder, confidence=0.5, static_image_mode=False, keypoint_folder=mediapipe_folder
        )

    # Symmetry constraint
    use_symmetry = args.sign_class in ("1a", "1b", "2a")

    print("Running SGNify...")
    run_sgnify(
        result_folder=result_folder,
        output_folder=output_folder,
        spectre_folder=spectre_folder,
        reference=reference,
        use_symmetry=use_symmetry,
        symmetry=symmetry,
        beta_path=beta_path,
        left_interp_folder=left_interp_folder,
        right_interp_folder=right_interp_folder,
        segment_path=segment_path,
    )

    # Create video with the results
    print("Creating video...")
    create_video(
        images_folder=output_folder.joinpath("images", "*.png"),
        output_path=output_folder.joinpath(f"{args.output_name}.mp4"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=False, default="None", help="Path to the video to analyze")
    parser.add_argument("--image_dir_path", required=False, default="None", help="Path to the image folder to analyze")
    parser.add_argument("--output_folder", required=False, type=str, default=".", help="Output folder")
    parser.add_argument("--output_name", required=False, type=str, default="output", help="Output name")
    parser.add_argument(
        "--sign_class",
        required=False,
        type=str,
        default="-1",
        choices=["-1", "0a", "0b", "1a", "1b", "2a", "2b", "3a", "3b"],
        help="Class of the sign",
    )
    parser.add_argument("--skip_preprocessing", required=False, action="store_true", help="Skip preprocessing of data")
    parser.add_argument("--quantitative_data", required=False, action="store_true", help="Analyze vicon frames")

    args = parser.parse_args()
    main(args)
