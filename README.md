This is the official code of GSNet: Generating 3D Garment Animation via Graph Skinning Network

This repository contains the necessary code 
![image](figure.gif)

<h3>DATA</h3>
The dataset used on this work and this repository is <a href="http://chalearnlap.cvc.uab.es/dataset/38/description/">CLOTH3D</a></a>.
<br>
Path to data has to be specified at 'values.py'. Note that it also asks for the path to preprocessings, described below.

<h4>PREPROCESSING</h4>
<a href="https://github.com/hbertiche/DeePSD/tree/master/Preprocessing">DeePSD</a>

<h3>SMPL</h3>
We removed SMPL models in PKL format due to their size. The code will expect those as '/DeePSD/Model/smpl/model_f.pkl' and '/DeePSD/Model/smpl/model_m.pkl'.
