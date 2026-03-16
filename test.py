import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)

y = 3 * X + 7 + np.random.normal(0, 1.5, len(X))

print("Dữ liệu:")
for xi, yi in zip(X,y):
    print(f"{xi:.0f} năm KN -> {yi:.1f} triệu")
    
X_mean = X.mean()
Y_mean = y.mean()

numerator = np.sum((X - X_mean) * (y - Y_mean))
denominator =  np.sum((X - X_mean) ** 2)

w = numerator / denominator
b =  Y_mean - w * X_mean

print(f"\nModel học được:")
print(f"  w (hệ số góc) ={w:.4f}")
print(f"  b (hệ số chặn) ={b:.4f}")
print(f"  → Phương trình: ŷ ={w:.2f}x +{b:.2f}")

y_pred = w * X + b

mse = np.mean((y_pred - y) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot =  np.sum((y - Y_mean) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\nĐánh giá model:")
print(f"  MSE ={mse:.4f}")
print(f"  R²  ={r2:.4f} ({r2*100:.1f}% variance explained)")

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Dữ liệu thực', zorder=5)
plt.plot(X, y_pred, color='red', linewidth=2, label=f'ŷ ={w:.2f}x +{b:.2f}')