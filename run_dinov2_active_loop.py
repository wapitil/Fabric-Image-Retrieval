import predict_gallery_with_classifier
import train_dinov2_classifier


def main():
    print("步骤 1/2: 训练 DINOv2 线性分类器")
    train_result = train_dinov2_classifier.run_training()
    print("\n步骤 2/2: 对全图库做自动预测和分流")
    predict_gallery_with_classifier.run_prediction(
        {
            "classifier_artifact_path": train_result["artifact_path"],
            "dataset_manifest_csv_path": train_result["manifest_path"],
        }
    )
    print("\n闭环主链路已完成。")


if __name__ == "__main__":
    main()
