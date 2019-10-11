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
git clone https://github.com/alldbi/SmoothFool.git
cd SmoothFool

# Generating smooth adversarial examples:
python smoothfool.py \
  --net resnet101 \
  --img "path to the input image" \
  --type "type of smoothing which can be gaussian, linear, or uniform." \
  --sigma "parameter of the smoothing function, for gaussian is the standard deviation, for linear and uniform is the size of kernel" \
  --smoothclip "whether using smoothclip or conventional clip" \
```


### Sample results

#### Gaussian

![](https://github.com/alldbi/SmoothFool/blob/master/samples/sample_gaussian.png)

#### Linear

![](https://github.com/alldbi/SmoothFool/blob/master/samples/sample_linear.png)

#### Uniform

![](https://github.com/alldbi/SmoothFool/blob/master/samples/samples_uniform.png)

## Citation
If you use the code or methodology for your research, please cite the paper: <a href="https://arxiv.org/abs/1910.03624">SmoothFool: An Efficient Framework for Computing Smooth Adversarial Perturbations</a>

## References
- [DeepFool](https://github.com/LTS4/DeepFool)
