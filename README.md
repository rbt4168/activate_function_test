# A small experiment of activate function

## Summary
This repository explores the impact of activation function complexity on neural network adaptation in PyTorch. The provided code introduces two custom activation functions, `MagicActivate` and `MagicActivate2`, and investigates their ability to adapt to complex functions compared to a standard ReLU activation.

## Conclusion
The observation that more complex activation functions lead to higher adaptation in handling complex functions aligns with the notion that expressive activation functions can enhance a neural network's ability to capture intricate patterns. However, it's essential to balance complexity to avoid overfitting.

## Result Preview

### The functions which ReLU performed better than ComplexActivate
![](/assets/relu_win/result_0.png)
![](/assets/relu_win/result_sin(x)cos(x).png)
![](/assets/relu_win/result_exp(-x2)sin(x).png)
### The functions which ComplexActivate is better
![](/assets/modified_win/result_tan(x)in0.2.png)
![](/assets/modified_win/result_exp(floor(x)+cos(x))-floor(x).png)
![](/assets/modified_win/result_fur.png)
