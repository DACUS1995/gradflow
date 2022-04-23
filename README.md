# A small, educational, numpy based deep learning framework with minimal PyTorch-like functionality.
![gradflow](assets/logo.png "gradflow")

---
This gradient computation engine uses the [reverse accumulation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation) method of automatic differentiation which involves calculating the gradient from the outermost operation inwards.