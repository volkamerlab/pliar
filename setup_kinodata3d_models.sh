# clone the kinodata-3d repo anc checkout out the pli-alignment-models branch
# clone direclty into this directory
TARGET_DIR=kinodata3d_models
mkdir -p $TARGET_DIR
git clone --branch kinodata-pli-alignment git@github.com:volkamerlab/kinodata-3D-affinity-prediction.git $TARGET_DIR
cd $TARGET_DIR
mkdir data
mkdir data/raw data/processed
cd ..
# kinodata3D raw data
uv run obtain_data.py https://zenodo.org/records/10852507 --target-dir $TARGET_DIR --single-file kinodata_3d.zip
# mapping between residue and atom indices
uv run obtain_data.py https://zenodo.org/records/19145842 --target-dir $TARGET_DIR --single-file residue_atom_index.zip
cd $TARGET_DIR
echo "README.md:"
cat README.md
echo "pli_alignment_scripts/README.md"
cat pli_alignment_scripts/README.md
