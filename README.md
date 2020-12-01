-- This is a project for my master's degree course, Learning from Images --

Participants: Me. Also, myself.

# LicensePlateReconstructor

This project aims to reconstruct readable and rectified license plates from pictures of the [CCPD dataset](https://github.com/detectRecog/CCPD). To this end we will be using a special kind of generative adversarial network (GAN) / just a very sophisticated loss. (is it still a GAN, if the discriminator is fixed?)

## Dataset

The CCPD dataset contains over 300k images of chinese parking cars in varying weather and light conditions. Additional features are location, rotation and content of the license plates, as well as a blurriness and a lighting score.

## The Discriminator

As a dataset with matching target images is missing, our discriminator will not differentiate between the reconstructed image and a target. Instead it will be an Optical Character Recognition (OCR) DNN. This OCR model will try to read the reconstructed license plate, providing its own loss as loss for the generator model.

The architecture for the OCR will most likely follow the CRNN as used in [PP-OCR: A Practical Ultra Lightweight OCR System](https://arxiv.org/abs/2009.09941v3) and [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717).

## The Generator

To force the generator model to generate an image that resembles a license plate as closely as possible we introduce an auxiliary loss: the error between a "template" and the generated image. This template can be seen as the mean of all possible license plate combinations. The error between this template and the generated image will serve as auxiliary loss. I did not yet find a paper that follows this approach.

The generator's architecture will follow [U-GAT-IT](https://arxiv.org/abs/1907.10830), an attention-based image-to-image model.
