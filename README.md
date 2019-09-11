# SmoothFool

Pytorch implementation of "SmoothFool: a framework for computing smooth adversarial perturbations".

## Setup

### Prerequisites
- Pytorch > 0.4
- Python 3.5
- PIL 
- Matplotlib
- Numpy

### Getting Started

```sh
# clone this repo
git clone https://github.com/alldbi/FLM.git
cd FLM

# Generating adversarial faces by Grouped FLM:
python main.py \
  --method GFLM \
  --pretrained_model "path to the Inception ResNet v1 model trained on CASIA-WebFace" \
  --dlib_model "path to the pretrained model of the Dlib landmark detector" \
  --img "path to the input image" \
  --label "label of the input image" \
  --output_dir "path to the directory to save results"
  --epsilon "coefficient for a scaling the gradient sign for each single iteration of the attack"

# Generating adversarial faces by FLM:
python main.py \
  --method GFLM \
  --pretrained_model "path to the Inception ResNet v1 model trained on CASIA-WebFace" \
  --dlib_model "path to the pretrained model of the Dlib landmark detector" \
  --img "path to the input image" \
  --label "label of the input image" \
  --output_dir "path to the directory to save results"
  --epsilon "coefficient for a scaling the gradient sign for each single iteration of the attack"
```


### Sample results

#### Gaussian

![](https://github.com/alldbi/SmoothFool/blob/master/samples/sample_gaussian.png)

#### Linear

![](https://github.com/alldbi/SmoothFool/blob/master/samples/samples_linear.png)

#### Uniform

![](https://github.com/alldbi/SmoothFool/blob/master/samples/samples_uniform.png)


## References
- [FaceNet](https://github.com/davidsandberg/facenet)
- [TensorFlow STN implementation](https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py)
