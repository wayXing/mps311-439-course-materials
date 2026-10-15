#%%
# ---- Try with AI 1.1 ----
grid_hp = np.linspace(46, 230, 100)                               # 100 horsepower values
grid_table = pd.DataFrame({"horsepower": grid_hp, "hp_power": grid_hp ** 2})
curve = model_q.predict(grid_table)                               # the model's prediction for each value
plt.scatter(X_train["horsepower"], y_train, s=12)                 # the training cars
plt.plot(grid_hp, curve, color="green")                           # the fitted curve
plt.xlabel("Horsepower")
plt.ylabel("mpg")
plt.show()
car_225 = pd.DataFrame({"horsepower": [225], "hp_power": [225 ** 2]})
print("prediction for 225 hp:", model_q.predict(car_225))
#%%
# ---- Try with AI 2.2 ----
train_list = []
test_list = []
for degree in range(1, 8):
    poly_d = PolynomialFeatures(degree=degree, include_bias=False)
    scaler_d = StandardScaler()
    H10 = poly_d.fit_transform(ten_X)
    Z10 = scaler_d.fit_transform(H10)
    Ht = poly_d.transform(X_test[["horsepower"]])
    Zt = scaler_d.transform(Ht)
    model_d = LinearRegression()
    model_d.fit(Z10, ten_y)
    pred_10 = model_d.predict(Z10)
    pred_t = model_d.predict(Zt)
    train_list.append(mean_squared_error(ten_y, pred_10) ** 0.5)    # store the training RMSE
    test_list.append(mean_squared_error(y_test, pred_t) ** 0.5)     # store the test RMSE
plt.plot(range(1, 8), train_list, marker="o", label="training RMSE (10 cars)")
plt.plot(range(1, 8), test_list, marker="o", label="test RMSE (79 cars)")
plt.yscale("log")
plt.xlabel("degree")
plt.ylabel("RMSE (mpg)")
plt.legend()
plt.show()
#%%
# ---- Try with AI 3.1 ----
plain = LinearRegression()
plain.fit(Z7_ten, ten_y)
positions = np.arange(7)
plt.bar(positions - 0.2, np.abs(plain.coef_), width=0.4, label="no penalty")
plt.bar(positions + 0.2, np.abs(ridge.coef_), width=0.4, label="Ridge")
plt.yscale("log")
plt.xlabel("coefficient number")
plt.ylabel("absolute size")
plt.legend()
plt.show()
print("largest Ridge coefficient:", np.abs(ridge.coef_).max())
#%%
# ---- Try with AI 4.3 ----
plt.plot(alphas, cv_rmse, marker="o")
plt.xscale("log")
plt.scatter([best_alpha_ridge], [min(cv_rmse)], color="red", zorder=3)
plt.xlabel("alpha")
plt.ylabel("CV RMSE (mpg)")
plt.show()
