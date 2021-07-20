pred_y = []
target_y = []

print("Predicting...")
for x in range(11):
    for y in range(11):
        #sp.show_process()
        area_X = test_x[:10, :, x:x+105, y:y+105]
        area_y = test_y[:10].reshape(-1, model_parameter['predict_period'], 11, 11)[:, :, x, y]
        target_y.append(area_y[:, :, np.newaxis])

        temp_y = []
        for idx, val in enumerate(area_X):
            classified = kmean_model.predict([val.flatten()])[0]
            model.load_weights(classified_model_pathes[classified])
            temp_y.append(model.predict(val[np.newaxis]))

        pred_y.append(np.concatenate(temp_y, 0)[:, :, np.newaxis])

test_y = np.concatenate(target_y, axis=2)
pred_y = np.maximum(np.concatenate(pred_y, 2), 0)
            

print(test_y.shape, pred_y.shape)