OWNER=local

for z in assets/models_zip/*.zip; do
    name=$(basename "$z" .zip)
    dest=/home/erich/Software/TexasStateUniversity-HiPE-Thesis/assets/models/fuel.ignitionrobotics.org/$OWNER/models/"$name"/1
    mkdir -p "$dest"
    unzip -o "$z" -d "$dest"
done
