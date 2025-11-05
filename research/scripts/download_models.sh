ROOT=$(python3 -c "from config import MODELS_PATH; print(MODELS_PATH)")
mkdir -p "$ROOT"
cd "$ROOT" || exit 1

echo "Downloading models to $ROOT ..."

# SuperPoint
curl -s -L -o sp_v6.tgz https://github.com/rpautrat/SuperPoint/raw/master/pretrained_models/sp_v6.tgz
tar -zxf sp_v6.tgz && rm sp_v6.tgz

# DINOv2 ViT-B/14
curl -s -L -o dinov2_vitb14_pretrain.pth  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
# OmniGlue export
curl -s -L -o og_export.zip https://storage.googleapis.com/omniglue/og_export.zip
unzip -qq og_export.zip && rm og_export.zip

echo "✅ Models ready in: $ROOT"