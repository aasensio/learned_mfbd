# Real-time multiframe blind deconvolution of solar images (submitted to Astronomy & Astrophysics)

## Introduction
The quality of images of the Sun obtained from the ground are severely
limited by the perturbing effect of the turbulent Earth's atmosphere. The post-facto correction
of the images to compensate for the presence of the atmosphere require the combination 
of high-order adaptive optics techniques,
fast measurements to freeze the turbulent atmosphere and very time consuming blind
deconvolution algorithms. Under mild seeing conditions, blind deconvolution
algorithms can produce images of astonishing quality. They can be very
competitive with those obtained from space, with the huge advantage of the flexibility
of the instrumentation thanks to the direct access to the telescope. In this
contribution we leverage deep learning techniques to significantly accelerate the
blind deconvolution process and produce corrected images at a peak rate of ~100 images
per second. We present two different architectures that produce excellent image
corrections with noise suppression while maintaining the photometric properties 
of the images. As a consequence, polarimetric signals can be obtained with
standard polarimetric modulation without any significant artifact.  
With the expected improvements in computer hardware and algorithms, we anticipate that
on-site real-time correction of solar images will be possible in the near future.

## Dependencies
This code depends on PyTorch >=0.4.0 but can be easily adapted to prior versions with
not much effort.

## Usage
We provide a simple example of a monochromatic H-alpha burst observation and two scripts
to use the two neural networks trained for this work. We note that observations need to
be multiplied by 0.001 before entering the networks and multiplied by 1000 after
the exit. The input should be given in counts as output by the data reduction pipeline
of CRISP@SST.

    python test_recurrent.py
    python test_encdec.py

## Training
We do not provide the training scripts because the data used for training 
is too heavy. They should be pretty straighforward to build following the
standard PyTorch training approach.