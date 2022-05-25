from Transparency.common_code.common import *
from Transparency.common_code.metrics import *
import Transparency.model.Question_Answering as QA
from sklearn.metrics import auc
import numpy as np

import pickle, os
from os.path import exists
class Trainer() :
    def __init__(self, dataset, config, _type='qa') :
        Model = QA.Model
        self.model = Model(config, pre_embed=dataset.vec.embeddings)
        self.metrics = calc_metrics_qa
        self.display_metrics = True
    
    def train(self, train_data, test_data, n_iters=20, save_on_metric='accuracy') :
        best_metric = 0.0
        for i in tqdm(range(n_iters)) :
            self.model.train(train_data)
            predictions, attentions = self.model.evaluate(test_data)
            predictions = np.array(predictions)
            test_metrics = self.metrics(test_data.A, predictions)
            if self.display_metrics :
                print_metrics(test_metrics)

            metric = test_metrics[save_on_metric]
            if metric > best_metric and i > 0 :
                best_metric = metric
                save_model = True
                print("Model Saved on ", save_on_metric, metric)
            else :
                save_model = False
                print("Model not saved on ", save_on_metric, metric)
            
            dirname = self.model.save_values(save_model=save_model)
            f = open(dirname + '/epoch.txt', 'a')
            f.write(str(test_metrics) + '\n')
            f.close()

class Evaluator() :
    def __init__(self, dataset, dirname, _type='qa') :
        Model = QA.Model
        self.model = Model.init_from_config(dirname)
        self.model.dirname = dirname
        self.metrics = calc_metrics_qa
        self.display_metrics = True

    def evaluate(self, test_data, save_results=False, return_metrics=False) :
        predictions, attentions = self.model.evaluate(test_data)
        predictions = np.array(predictions)

        test_metrics = self.metrics(test_data.A, predictions)
        if self.display_metrics :
            print_metrics(test_metrics)

        if save_results :
            f = open(self.model.dirname + '/evaluate.json', 'w')
            json.dump(test_metrics, f)
            f.close()

        test_data.yt_hat = predictions
        test_data.attn_hat = attentions
        if not return_metrics:
            return predictions, attentions
        else:
            return predictions, attentions, test_metrics
    def permutation_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'permutations') :
            perms = self.model.permute_attn(test_data)
            pdump(self.model, perms, 'permutations')

    def adversarial_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'multi_adversarial') :
            multi_adversarial_outputs = self.model.adversarial_multi(test_data)
            pdump(self.model, multi_adversarial_outputs, 'multi_adversarial')

    def remove_and_run_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'remove_and_run') :
            remove_outputs = self.model.remove_and_run(test_data)
            pdump(self.model, remove_outputs, 'remove_and_run')

    def gradient_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'gradients') :
            grads = self.model.gradient_mem(test_data)
            pdump(self.model, grads, 'gradients')


    def replace_and_run_experiment(self, model_name, test_data, exp="att", delta_type="att_mask", exp_metric="AUCTP", mask_size=1,
                                   plot=False, load_exp_scores=False, load_perturbation_scores=True, verbose=True):
        assert exp_metric in ["AUCTP", "Sufficiency", "Comprehensiveness", "Correlation", "Violation", "RC"]

        dir_path = './data/qa'
        save_scores_path = os.path.join(dir_path, model_name, exp)
        create_dir(save_scores_path)
        print(save_scores_path)
        file_name = exp_metric+'_'+delta_type +'.pkl'
        if load_perturbation_scores and exists(os.path.join(save_scores_path, file_name)):
            with open(os.path.join(save_scores_path, file_name), 'rb') as f:
                outputs = pickle.load(f)
                top_ans_deltas = torch.from_numpy(outputs['top_ans_deltas'])
                all_pred_deltas = torch.from_numpy(outputs['all_pred_deltas'])
                preds = outputs['preds']
                new_preds = outputs['new_preds']
                weights = torch.from_numpy(outputs['weights'])
                mask_ratios = outputs['mask_ratios']
                Y = outputs['label']
        else:
            preds, new_preds, weights, top_ans_deltas, all_pred_deltas, mask_ratios = \
                self.model.replace_and_run(test_data, exp, delta_type,
                                           exp_metric, 20, mask_size,
                                           load_exp_scores=load_exp_scores)
            Y = test_data.A
            with open(os.path.join(save_scores_path, file_name), 'wb') as f:
                print("save {} perturbation scores to {}".format(file_name, save_scores_path))
                pickle.dump({
                    "top_ans_deltas": top_ans_deltas.numpy(),
                    "new_preds": new_preds,
                    "all_pred_deltas": all_pred_deltas.numpy(),
                    "preds": preds,
                    "weights": weights.numpy(),
                    "label": test_data.A,
                    "mask_ratios": mask_ratios,
                }, f)


        overall_test_metrics = self.metrics(Y, preds)
        exp_values = {}

        if exp_metric == "AUCTP":
            # Calculate Comprehensiveness Here
            comprehensiveness = top_ans_deltas.squeeze().mean(dim=0) # [N, K] -> [K]
            ratio_index = [0,1,2,5] # [5%, 10%, 20%, 50%]
            print(mask_ratios[ratio_index])
            comprehensiveness = comprehensiveness[ratio_index].mean()
            exp_values["Comprehensiveness"] = comprehensiveness.item()

            # Calculate AUCTP Here (https://github.com/copenlu/xai-benchmark)
            # eval_key = 'macro avg/f1-score' if self.metric_type == 'Single_Label' else 'accuracy'
            eval_key = 'accuracy'
            scores = [overall_test_metrics[eval_key]]
            for i in range(len(mask_ratios)):
                test_metrics = self.metrics(Y, new_preds[:, i])
                scores.append(test_metrics[eval_key]) # [0%, 5%, 10%, 20%, 30%, ..., 90%]

            mask_ratios = np.insert(mask_ratios, 0, 0)
            exp_values["AUCTP"] = auc(mask_ratios, scores)
            if verbose:
                print(exp_values)
                print(mask_ratios, scores)

            if plot:
                return {"Comprehensiveness": top_ans_deltas[:, ratio_index].mean(1).numpy()}, exp_values

        elif exp_metric in ["Sufficiency", "Comprehensiveness"]:
            if verbose:
                print("Calculate {} for {}".format(exp_metric, mask_ratios))
            scores = top_ans_deltas.squeeze().mean(dim=0) # eg, [5%, 10%, 20%, 50%]
            exp_values[exp_metric] = scores.mean().item()
            if plot:
                return {exp_metric: top_ans_deltas.mean(1).numpy()}, exp_values

        elif exp_metric == "Violation":
            vio_ind = calculate_violators(top_ans_deltas, weights)
            VioRatio = vio_ind.sum() / top_ans_deltas.shape[0]
            exp_values["Violation"] = VioRatio.item()
            if verbose:
                print("num. of violators: {}, ratio:{:.2%}".format(vio_ind.sum(), VioRatio))
            if plot:
                return {"Violation": vio_ind, "weights": weights,
                        "top_ans_deltas": top_ans_deltas.numpy()}, exp_values

        elif exp_metric == "RC":
            RCC = compute_rank_correlation(weights.abs(), all_pred_deltas.abs())
            print(" rcc:{}".format(RCC.mean()))
            exp_values["RC"] = RCC.mean().item()
            if plot:
                return {"RC": RCC}, exp_values

        return preds, new_preds, weights, top_ans_deltas, all_pred_deltas, exp_values