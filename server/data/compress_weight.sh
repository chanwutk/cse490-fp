tar -cvvzf ./weights.tar.gz ./weights.pt
split -b 50m ./weights.tar.gz "weights_small_"
rm -f ./weights.tar.gz
