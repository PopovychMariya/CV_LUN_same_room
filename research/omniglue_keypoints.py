from tqdm import tqdm
import omniglue
import numpy as np
import torch
import os


def save_keypoints(loader, path, og):
    os.makedirs(path, exist_ok=True)
    for batch in tqdm(loader, desc=f"Loading batches"):
        for idx, (img1, img2) in enumerate(zip(batch["image1"], batch["image2"])):
            kp1, kp2, conf = og.FindMatches(np.array(img1), np.array(img2))
            task_id = batch["task_id"][idx]
            H1, W1 = img1.shape[:2]
            H2, W2 = img2.shape[:2]

            out_fp = path / task_id
            os.makedirs(out_fp, exist_ok=True)

            np.savez_compressed(
                out_fp / "matches.npz",
                kp1=kp1.astype(np.float32),
                kp2=kp2.astype(np.float32),
                conf=conf.astype(np.float32),
                H1=np.int32(H1), W1=np.int32(W1),
                H2=np.int32(H2), W2=np.int32(W2),
            )


if __name__ == "__main__":
    import torch
    from dataloader import make_loader
    from path_config import (
        DATASET_FOLDER_PATHS,
        DATASET_ANNOTATIONS,
        MODELS_PATH,
        OMNIGLUE_KEYPOINTS,
    )

    print("Preparing dataloaders...")
    train_ds, train_loader = make_loader(
        annotation_path=DATASET_ANNOTATIONS["train_annotation_path"],
        dataset_path=DATASET_FOLDER_PATHS["train_folder_path"],
        mode="train", batch_size=64, num_workers=8, shuffle=False
    )
    test_ds, test_loader = make_loader(
        annotation_path=DATASET_ANNOTATIONS["test_annotation_path"],
        dataset_path=DATASET_FOLDER_PATHS["test_folder_path"],
        mode="test", batch_size=64, num_workers=8, shuffle=False
    )

    torch.multiprocessing.set_start_method("spawn", force=True)
    print("Loading OmniGlue weights...")
    og = omniglue.OmniGlue(
        og_export=str(MODELS_PATH / "og_export"),
        sp_export=str(MODELS_PATH / "sp_v6"),
        dino_export=str(MODELS_PATH / "dinov2_vitb14_pretrain.pth"),
    )

    print("Extracting keypoints for train set...")
    save_keypoints(train_loader, OMNIGLUE_KEYPOINTS["train"], og)
    print("Extracting keypoints for test set...")
    save_keypoints(test_loader, OMNIGLUE_KEYPOINTS["test"], og)
    print("✅ All keypoints detected and saved successfully.")
    
