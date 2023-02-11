# Molecular Graph AutoEncoder

This code uses the NFP Library to convert SMILE strings into graph molecular structures. From there we attempt to compress these molecular graphs.


# Useful links
1. https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py
2. https://github.com/CEGRcode/Enhanced_Transformer_For_Enhancers/blob/master/Model/main.py

# Resources

1. Download the container: docker build -t <tag_name> .
2. Run the container: docker run -it --rm --gpus all --name pytorch -v $PWD:/work <tag_name> 
3. upon starting the docker script you need to install pyg (gpu or cpu auto detected) with the command: conda install pyg -c pyg
