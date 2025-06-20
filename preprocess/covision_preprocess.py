import h5py
import json
import os

def generate_h5_file(output_path, num_sequences, base_path):
    """
    Generate dps_train.h5 file with the specified structure.

    Args:
        output_path (str): Path to save the generated H5 file.
        num_sequences (int): Number of sequences to generate.
        base_path (str): Base path for the data files.
    """
    sequences = []

    for i in range(num_sequences):
        scene_name = f"scene{i:04d}_00"

        # Generate paths for rgb, depth, and pose lists
        rgb_list = [
            os.path.join(base_path, scene_name, "frames/color", f"{j}.jpg")
            for j in range(661, 671)
        ]
        rgb_list = rgb_list + [rgb_list[-1]] * (30 - len(rgb_list))

        depth_list = [
            os.path.join(base_path, scene_name, "frames/depth", f"{j}.png")
            for j in range(661, 671)
        ]
        depth_list = depth_list + [depth_list[-1]] * (30 - len(depth_list))

        pose_list = [
            os.path.join(base_path, scene_name, "frames/pose", f"{j}.txt")
            for j in range(661, 671)
        ]
        pose_list = pose_list + [pose_list[-1]] * (30 - len(pose_list))

        # Intrinsic matrix
        intrinsic_raw = [
            [577.870605, 0.0, 319.5, 0.0],
            [0.0, 577.870605, 239.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        # Combine into a sequence dictionary
        sequence = {
            "scene_name": scene_name,
            "rgb_list": rgb_list,
            "depth_list": depth_list,
            "pose_list": pose_list,
            "intrinsic_raw": intrinsic_raw,
        }

        # Convert to JSON string and append to the list
        sequences.append(json.dumps(sequence))

    # Write sequences to H5 file
    with h5py.File(output_path, "w") as h5_file:
        json_strs = h5_file.create_dataset("json_strs", data=[s.encode("utf-8") for s in sequences])
        print(f"H5 file created at {output_path} with {len(json_strs)} sequences.")

# Example usage
generate_h5_file(
    output_path="../trajectories/covision_train/dps_train.h5",
    num_sequences=7223,  # Adjust this value based on your requirement
    base_path="data/scannet"
)


