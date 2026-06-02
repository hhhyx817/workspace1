forgery detection has D_for_tr, D_for_cal, D_for_test
the algo divides every pixel of D_for_test picture into "forgery" and "not forgery"
obtain F1 and AUC from D_for__test




for image: F1 = 2 × (Precision + Recall)/(Precision × Recall)
           precision = forgeried pixel / predicted forgeried pixel (in a picture)
​           recall = predicted forgeried pixel / forgeried pixel
for D_for_test: F1 = avg(F1 of image)

in function: Inference_loc() 
for img_idx, cur_img_name in enumerate(image_names):
    mask_ = mask[img_idx].cpu().numpy().reshape(-1)
    pred_mask_ = pred_mask[img_idx].cpu().numpy().reshape(-1)

    F1_a = metrics.f1_score(mask_, pred_mask_, average='macro')
    # ( set True = forgeried / not forgeried, F1 maight be different when "1" and "0" are unbalanced)
    pred_mask_[np.where(pred_mask_ == 0)] = 1
    pred_mask_[np.where(pred_mask_ == 1)] = 0
    F1_b = metrics.f1_score(mask_, pred_mask_, average='macro')

    F1 = max(F1_a, F1_b)
    F1_lst.append(F1)
print("F1: ", np.mean(F1_lst))




the CP algo should focus on D_for_test s.t. P(every pixel of D_for_test picture predictd to be "forgery" is "forgery") >= 0.9
$ 不用考虑 D_for_tr, D_for_cal (forgery 算法有没有 cal 都没关系)
$ considering exchageable case: 
  D_cp_cal = {forgeried pixel in D_for_tr, D_for_cal}, NCM S = 1 - P(the pixed is predicted forgery), form {NCM set}
  D_cp_test = D_for_test, if 1 - P(predicted forgery) <= Q(NCM set, 0.9), then label the pixel as forgeried

