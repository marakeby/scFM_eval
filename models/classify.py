import pandas as pd
from os.path import join
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import seaborn as sns
from matplotlib import pyplot as plt

from evaluation.eval import eval_classifier, plot_classifier
from run.run_utils import get_split_dict
from utils.saving import save_supervised
from utils.logs_ import get_logger
from models.mil_experiment import MILExperiment
# from models.gf_finetune import GFFineTuneModel
import numpy as np


logger = get_logger()

MODEL_REGISTRY = {
    'random_forest': RandomForestClassifier,
    'logistic_regression': LogisticRegression,
    'svc': SVC
    # 'gf_finetune': GFFineTuneModel
}

def save_results(pred_df, metrics_df, cls_report, saving_dir, postfix, viz=False, model_name=None, label_names=None):
    """Utility function to save results and visualizations"""
    
    if pred_df is not None:
        fname = join(saving_dir, f'cls_predictions_{postfix}.csv')
        pred_df.to_csv(fname)
    
    if metrics_df is not None:
        fname = join(saving_dir, f'cls_metrics_{postfix}.csv')
        metrics_df.to_csv(fname)
    
    if cls_report is not None:
        fname = join(saving_dir, f'cls_report_{postfix}.csv')
        pd.DataFrame(cls_report).transpose().to_csv(fname)
    
    if viz and pred_df is not None:
        fig = plot_classifier(pred_df['label'], pred_df['pred'], pred_df['pred_score'], estimator_name=model_name, label_names=label_names)
        fname = join(saving_dir, f'cls_metrics_{postfix}.png')
        fig.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()

def train_classifier(X_train, y_train, X_test, y_test, model_name='random_forest'):
    """Train a classifier and return predictions"""
    model_cls = MODEL_REGISTRY.get(model_name, RandomForestClassifier)
    model = model_cls(probability=True) if model_name == 'svc' else model_cls()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_score_test = model.predict_proba(X_test)
    
    y_pred_train = model.predict(X_train)
    y_pred_score_train = model.predict_proba(X_train)
    
    # return model, y_test, y_pred, y_pred_score
    return model, y_train, y_test, y_pred_train, y_pred_test, y_pred_score_train, y_pred_score_test

class ClassifierPipeline:
    def __init__(self, params):  
        self.params = params['params']
        logger.info(f'ClassifierPipeline {params}')
        
        self.saving_dir = self.params['save_dir']
        self.model_name = self.params['model']
        # self.model_dir = self.params['model_dir']
        self.viz = params['viz']
        self.evaluate = params['eval']
        self.embedding_col = self.params['embedding_col']
        self.label_map = self.params['label_map'] #dictionary e.g. # {'Pre': 0, 'Post': 1}
        self.cv = self.params.get('cv', False)
        self.onesplit = self.params.get('onesplit', False)
        self.cls_level = self.params.get('cls_level', 'patient')
        self.train_funcs = self.params['train_funcs']
        self.model = None
        # self.label_encoder = LabelEncoder()
        self.label_names = None

    def encode_labels(self):
        """Encode labels using LabelEncoder"""
        # self.adata.obs['label'] = self.label_encoder.fit_transform(self.adata.obs[self.label_key])
        self.adata.obs['label'] = self.adata.obs[self.label_key].map(self.label_map) # {'Pre': 0, 'Post': 1}
        # self.label_names = list(self.label_encoder.classes_)
        self.label_names = list(self.label_map.keys())
        logger.info(f"Label classes: {self.label_names}")

    def get_splits_cv(self):
        """Get cross-validation splits"""
        cv = self.data_loader.cv_split_dict
        n_splits = cv['n_splits']
        id_column = cv['id_column']
        
        train_ids_list = []
        test_ids_list = []
        for i in range(n_splits):
            train_ids = cv[f'fold_{i+1}']['train_ids']
            test_ids = cv[f'fold_{i+1}']['test_ids']
            train_ids_list.append(train_ids)
            test_ids_list.append(test_ids)
        return id_column, n_splits, train_ids_list, test_ids_list

    def split_data(self, id_column, train_ids, test_ids):
        """Split data into train and test sets"""
        adata_test = self.adata[self.adata.obs[id_column].isin(test_ids)]
        adata_train = self.adata[self.adata.obs[id_column].isin(train_ids)]
        return adata_train, adata_test

    def get_split_data(self):
        """Get train-test split data"""
        if hasattr(self.data_loader, 'train_test_split_dict'):
            split_dict = self.data_loader.train_test_split_dict
            test_ids = split_dict['train_test_split']['test_ids']
            train_ids = split_dict['train_test_split']['train_ids']
            id_column = split_dict['id_column']
            logger.info(test_ids)
        return id_column, train_ids, test_ids

    def prepare_data(self, adata_train, adata_test, id_column):
        """Prepare data for training"""
        adata_train.obs['sample_id'] = adata_train.obs[id_column]
        adata_test.obs['sample_id'] = adata_test.obs[id_column]
        return adata_train, adata_test

    def train(self, loader):
        self.data_loader = loader
        self.adata = loader.adata
        
        if self.cls_level == 'patient':
            self.train_sample(loader)
        elif self.cls_level =='cell':
            # Single split training
            id_column, train_ids, test_ids = self.get_split_data()
            adata_train, adata_test = self.split_data(id_column, train_ids, test_ids)
            adata_train, adata_test = self.prepare_data(adata_train, adata_test, id_column)
            self.__train_cell(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
        
    def train_sample(self, loader):
        """Main training pipeline"""
        
        self.label_key = loader.label_key
        self.encode_labels()
        train_func_map = {'vote': self.__train_vote, 'avg': self.__train_avg_expression, 'mil': self.__train_mil}
        train_fcs = [] 
        prefix =[]
        for fnc in self.train_funcs:
            train_fcs.append(train_func_map[fnc])
            prefix.append(fnc)
            
        # Single split training
        if self.onesplit:
            id_column, train_ids, test_ids = self.get_split_data()
            adata_train, adata_test = self.split_data(id_column, train_ids, test_ids)
            adata_train, adata_test = self.prepare_data(adata_train, adata_test, id_column)
            for fnc in train_fcs:
                fnc(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
                # self.__train_mil(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
                # self.__train_vote(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
                # self.__train_avg_expression(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
        
        # Cross-validation training
        if self.cv:
            id_column, n_splits, train_ids_list, test_ids_list = self.get_splits_cv()
            for tr_f, pre in zip(train_fcs, prefix):
                self.__train_cv(tr_f, id_column, n_splits, train_ids_list, test_ids_list, pre)

    def __train_cv(self, train_fnc, id_column, n_splits, train_ids_list, test_ids_list, prefix=''):
        """Run cross-validation training"""
        pred_list = []
        metrics_list = []
        pred_list_train = []
        metrics_list_train = []
        logger.info(f'Running crossvalidation with {n_splits} folds')
        
        for i in range(n_splits):
            logger.info(f'---------- fold {i+1}----------')
            train_ids, test_ids = train_ids_list[i], test_ids_list[i]
            adata_train, adata_test = self.split_data(id_column, train_ids, test_ids)
            adata_train, adata_test = self.prepare_data(adata_train, adata_test, id_column)
            
            # pred_df, metric_df = train_fnc(adata_train, adata_test, f'fold_{i+1}_')
            # pred_df_train, pred_df_test, metrics_df_train, metrics_df_test = train_fnc(adata_train, adata_test, f'fold_{i+1}_True
            pred_df_train, pred_df_test, metrics_df_train, metrics_df_test = train_fnc(adata_train, adata_test, evaluate=False, viz=False)

            pred_df_test['fold'] = f'fold_{i+1}'
            metrics_df_test['fold'] = f'fold_{i+1}'
            
            pred_df_train['fold'] = f'fold_{i+1}'
            metrics_df_train['fold'] = f'fold_{i+1}'
            
            pred_list.append(pred_df_test)
            metrics_list.append(metrics_df_test)
            
            pred_list_train.append(pred_df_train)
            metrics_list_train.append(metrics_df_train)
        
        preds = pd.concat(pred_list)
        metrics = pd.concat(metrics_list)
        
        preds_train = pd.concat(pred_list_train)
        metrics_train = pd.concat(metrics_list_train)
        
        save_dir = join(self.saving_dir, 'cv')
        os.makedirs(save_dir, exist_ok=True)
        
        preds.to_csv(join(save_dir, f'{prefix}_cv_predictions.csv'))
        metrics.to_csv(join(save_dir, f'{prefix}_cv_metrics.csv'))
        
        preds_train.to_csv(join(save_dir, f'{prefix}_cv_predictions_train.csv'))
        metrics_train.to_csv(join(save_dir, f'{prefix}_cv_metrics_train.csv'))
        
        # Plot metrics
        metrics.fillna(0, inplace=True)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Metrics', y=self.model_name, data=metrics)
        plt.title('Cross-Validation Metric Distribution')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(join(save_dir, f'{prefix}_cv_metrics_boxplot.png'))
        plt.close()
        
        metrics_train.fillna(0, inplace=True)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Metrics', y=self.model_name, data=metrics_train)
        plt.title('Cross-Validation Metric Distribution on Training set')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(join(save_dir, f'{prefix}_cv_metrics_boxplot_train.png'))
        plt.close()
        
        
        return preds, metrics

    def __train_avg_expression(self, adata_train, adata_test, evaluate=False, viz=False):
        """Train and evaluate using average expression per sample"""
        logger.info('Training model (Average Embedding Per Sample)')

        # Average embedding per sample
#         def aggregate_embeddings(adata):
#             emb = adata.obsm[self.embedding_col]
            
#             sample_ids = list(adata.obs['sample_id'].values)
#             print(sample_ids)
#             df_emb = pd.DataFrame(emb)
#             df_emb['sample_id'] = sample_ids
#             print(df_emb.head())
#             mean_emb = df_emb.groupby('sample_id').mean()
#             # mean_emb = df_emb.groupby(df_emb.index).mean()
            
#             labels = adata.obs[['sample_id', 'label']].drop_duplicates().set_index('sample_id')
#             return mean_emb.loc[labels.index], labels['label']
        def aggregate_embeddings(adata):
            emb = adata.obsm[self.embedding_col]

            # Convert sparse to dense if necessary
            if not isinstance(emb, np.ndarray):
                emb = emb.toarray()

            sample_ids = adata.obs['sample_id'].values
            df_emb = pd.DataFrame(emb, index=sample_ids)

            mean_emb = df_emb.groupby(df_emb.index).mean()

            labels = adata.obs[['sample_id', 'label']].drop_duplicates().set_index('sample_id')
            labels = labels.loc[labels.index.intersection(mean_emb.index)]

            return mean_emb.loc[labels.index], labels['label']


        X_train, y_train = aggregate_embeddings(adata_train)
        X_test, y_test = aggregate_embeddings(adata_test)
        

        # Train classifier
        self.model, y_train, y_test, y_pred_train, y_pred_test, y_pred_score_train, y_pred_score_test = train_classifier(X_train, y_train, X_test, y_test, model_name=self.model_name)

        pred_df_test = pd.DataFrame({ 'label': y_test,'pred': y_pred_test,
                                'pred_score': y_pred_score_test[:, 1]  # Assuming binary classification
                               }, index=X_test.index)

        pred_df_train = pd.DataFrame({ 'label': y_train,'pred': y_pred_train,
                                'pred_score': y_pred_score_train[:, 1]  # Assuming binary classification
                               }, index=X_train.index)
            
        metrics_df_test, cls_report_test = eval_classifier(pred_df_test['label'], pred_df_test['pred'], pred_df_test['pred_score'],
                                                 estimator_name=self.model_name, label_names=self.label_names)
        
        metrics_df_train, cls_report_train = eval_classifier(pred_df_train['label'], pred_df_train['pred'], pred_df_train['pred_score'],
                                                 estimator_name=self.model_name, label_names=self.label_names)

        # metrics_df_train= None
        # metrics_df_test = None
        if evaluate:
            postfix = "avg_expr"
            save_results(pred_df_test, metrics_df_test, cls_report_test, self.saving_dir, postfix, viz, self.model_name, self.label_names)
            postfix = "avg_expr_train"
            save_results(metrics_df_train, metrics_df_train, cls_report_train, self.saving_dir, postfix, viz, self.model_name, self.label_names)

        return pred_df_train, pred_df_test, metrics_df_train, metrics_df_test

    def __train_mil(self, adata_train, adata_test, evaluate=False, viz=False):
        """Multi-instance learning training"""
        logger.info('Training model (Multi instance Learning (MIL))')
        
        exp = MILExperiment(embedding_col=self.embedding_col, label_key='label', 
                          patient_key="sample_id", celltype_key="cell_type")
        exp.train(adata_train)
        
        #test
        pids, y_true, preds, pred_scores = exp.evaluate(adata_test)
        pred_df_test = pd.DataFrame({'id': pids, 'label': y_true, 'pred': preds, 'pred_score': pred_scores})
        #train
        pids, y_true, preds, pred_scores = exp.evaluate(adata_train)
        pred_df_train = pd.DataFrame({'id': pids, 'label': y_true, 'pred': preds, 'pred_score': pred_scores})
        metrics_df_test, cls_report_test = eval_classifier(pred_df_test['label'], pred_df_test['pred'], pred_df_test['pred_score'], estimator_name=self.model_name, label_names=self.label_names)
        
        metrics_df_train, cls_report_train = eval_classifier(pred_df_train['label'], pred_df_train['pred'], pred_df_train['pred_score'], estimator_name=self.model_name, label_names=self.label_names)
        
        if evaluate:

            
            save_results(pred_df_test, metrics_df_test, cls_report_test, self.saving_dir, postfix='mil', viz=viz, model_name=self.model_name, label_names=self.label_names)
            
            
            save_results(pred_df_train, metrics_df_train, cls_report_train, self.saving_dir, postfix='mil_train', viz=viz, model_name=self.model_name, label_names=self.label_names)
            
          
        # return pred_df, metrics_df if evaluate else None
        return pred_df_train, pred_df_test, metrics_df_train, metrics_df_test

    def __train_cell(self, adata_train, adata_test, evaluate=False, viz=False):
        """Cell-level predictions training"""
        logger.info('Training model')
        X_train, y_train = adata_train.obsm[self.embedding_col], adata_train.obs['label']
        X_test, y_test = adata_test.obsm[self.embedding_col], adata_test.obs['label']

        # self.model, y_test, y_pred, y_pred_score = train_classifier(X_train, y_train, X_test, y_test, 
        #                                                           model_name=self.model_name)
        self.model, y_train, y_test, y_pred_train, y_pred_test, y_pred_score_train, y_pred_score_test = train_classifier(X_train, y_train, X_test, y_test, 
                                                                  model_name=self.model_name)
        
        
        
        adata_test.obs['pred'] = y_pred_test
        adata_test.obs['pred_score'] = y_pred_score_test[:, 1] #assume binary classification
        
        adata_train.obs['pred'] = y_pred_train
        adata_train.obs['pred_score'] = y_pred_score_train[:, 1] #assume binary classification
        
        if evaluate:
            save_dir = join(self.saving_dir, 'cell_level_pred')
            os.makedirs(save_dir, exist_ok=True)
            
            adata_test.obs.to_csv(join(save_dir, f'cell_pred_test.csv'))
            adata_train.obs.to_csv(join(save_dir, f'cell_pred_train.csv'))
            
            metrics_df_test, cls_report = eval_classifier(y_test, y_pred_test, y_pred_score_test, estimator_name=self.model_name, label_names=self.label_names)
            pred_df = adata_test.obs['pred'].copy()
            save_results(pred_df, metrics_df_test,cls_report,save_dir, 'cell', viz, self.model_name, self.label_names)
            
            metrics_df_train, cls_report = eval_classifier(y_test, y_pred_test, y_pred_score_test, estimator_name=self.model_name, label_names=self.label_names)
            pred_df = adata_train.obs['pred'].copy()
            save_results(pred_df, metrics_df_train,cls_report,save_dir, 'cell_train', viz, self.model_name, self.label_names)
            
            # return adata_test.obs, metrics_df
            return adata_train.obs, adata_test.obs, metrics_df_train, metrics_df_test
        
        return adata_test.obs, None

    def __train_vote(self, adata_train, adata_test, evaluate=False, viz=False):
        """Majority vote predictions training"""
        logger.info('Training model (Majority Vote)')
        
        cell_pred_train, cell_pred_test, metrics_df_train, metrics_df_test = self.__train_cell(adata_train, adata_test, 
                                                             evaluate=evaluate, viz=viz)
        
        pred_df_test, metrics_df_test = self.save_patient_level(cell_pred_test, 
                                                                            evaluate, viz, 
                                                                            postfix='', 
                                                                            model='vote')
        
        pred_df_train, metrics_df_train = self.save_patient_level(cell_pred_train, 
                                                                            evaluate, viz, 
                                                                            postfix='_train', 
                                                                            model='vote')
        
        # return sample_pred_test_df, sample_metrics_test_df
        return pred_df_train, pred_df_test, metrics_df_train, metrics_df_test

    def save_patient_level(self, adata_subset, evaluate=False, viz=False, postfix="", model=""):
        """Save patient-level predictions and metrics"""
        logger.info('Saving sample level performance')
        
        obs = adata_subset
        
        def majority_vote_score(x):
            majority_class = x.value_counts().idxmax()
            return (x == majority_class).sum() / len(x)

        y_pred_score_p = obs.groupby('sample_id')['pred_score'].mean()
        
        # y_pred_score_p = obs.groupby('sample_id')['pred'].agg(majority_vote_score)
        
        y_pred_p = obs.groupby('sample_id')['pred'].agg(lambda x: x.value_counts().idxmax())
        y_test_p = obs.groupby('sample_id')['label'].first().reindex(y_pred_score_p.index)
        
        pred_df = pd.concat([y_test_p, y_pred_p, y_pred_score_p], axis=1)
        pred_df.columns = ['label', 'pred', 'pred_score']
        
        metrics_df, cls_report = eval_classifier(pred_df['label'], pred_df['pred'], pred_df['pred_score'],
                                              estimator_name=self.model_name, label_names=self.label_names)
        
        if evaluate:
            save_results(pred_df, metrics_df,cls_report, self.saving_dir, f'vote{postfix}', viz, self.model_name, self.label_names)
        
        return pred_df, metrics_df
