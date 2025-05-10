# 🏎️ PPO CarRacing-v2 Agent  
🧠 **Mastering Precision Driving with PPO**  
> A Proximal Policy Optimization (PPO) agent trained to race in `CarRacing-v2`, leveraging visual input through convolutional policies. Built using **Stable-Baselines3**, it learns to steer, brake, and accelerate across procedurally generated tracks.

---

## 🎥 Demo  


https://github.com/user-attachments/assets/fae225e6-207c-44f2-bc49-b86d33e8b7e8


---

## 🛠️ Features
- Uses `CnnPolicy` to handle image-based observations  
- Supports training with live environment rendering  
- Model checkpoint saving after each training episode  
- Smooth evaluation loop with average reward tracking  
- TensorBoard logging for monitoring PPO training progress

---

## 📦 Tech Stack
**Language**: Python
- **Stable-Baselines3**
- **Gymnasium**
- **PyTorch**
