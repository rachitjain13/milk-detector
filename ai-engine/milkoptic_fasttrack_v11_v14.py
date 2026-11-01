from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent
# Accept files beside script, or in the original project data location.
def locate(name):
    aliases = {'unified_all_batches(1).csv':['unified_all_batches.csv','unified_all_batches(1).csv'],
               'v6_roi_features(1).csv':['v6_roi_features.csv','v6_roi_features(1).csv']}
    candidates = [BASE/n for n in aliases.get(name,[name])] + [BASE/'research_data'/n for n in aliases.get(name,[name])] + [BASE/'analysis_output'/n for n in aliases.get(name,[name])]
    for p in candidates:
        if p.exists(): return p
    raise FileNotFoundError(f"Cannot find {name}. Put it beside this script or in research_data/.")

full_path = locate('unified_all_batches(1).csv')
roi_path = locate('v6_roi_features(1).csv')
out = BASE/'analysis_output'/'milkoptic_v11_v14_combined'
out.mkdir(parents=True, exist_ok=True)
full = pd.read_csv(full_path)
roi = pd.read_csv(roi_path)

# Basic integrity checks
for df, label in [(full,'unified'),(roi,'ROI')]:
    required={'batch','file_name','dilution_percent'}
    missing=required-set(df.columns)
    if missing: raise ValueError(f'{label} CSV missing columns: {missing}')
    if df['dilution_percent'].isna().any(): raise ValueError(f'{label} has missing targets')

# Feature columns are numeric only; exclude labels, identifiers, paths and ROI metadata.
exclude={'dilution_percent','concentration','batch','file_name','relative_path','file_path','camera_mode','format','zoom','roi_size'}
def feature_cols(df):
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

full_features=feature_cols(full)
roi_features=feature_cols(roi)

# Align the ROI rows by exact file identity and evaluate each ROI size independently.
# One row per image per feature set; GroupKFold additionally groups by filename.
sets={'FULL_IMAGE':(full,full_features)}
for size in sorted(roi['roi_size'].dropna().unique()):
    sub=roi[roi.roi_size==size].copy()
    sets[f'ROI_{int(size)}']=(sub,roi_features)

models={
    'Ridge_scaled': make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=10.0)),
    'ExtraTrees_leaf2': make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=150,min_samples_leaf=2,max_features=1.0,n_jobs=-1,random_state=42)),
    'RandomForest_leaf2': make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=150,min_samples_leaf=2,max_features=1.0,n_jobs=-1,random_state=42)),
    'HistGradientBoosting': make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=120,learning_rate=0.05,l2_regularization=1.0,random_state=42)),
}

rows=[]
oof_frames=[]
for set_name,(df,cols) in sets.items():
    b1=df[df.batch=='B1'].copy().reset_index(drop=True)
    X=b1[cols].replace([np.inf,-np.inf],np.nan)
    y=b1.dilution_percent.astype(float)
    groups=b1.dilution_percent.astype(str) + '_zoom_' + b1.zoom.astype(str)
    cv=GroupKFold(n_splits=3)
    for model_name,model in models.items():
        pred=cross_val_predict(model,X,y,groups=groups,cv=cv,n_jobs=1)
        rows.append({'feature_set':set_name,'model':model_name,'n':len(y),'features':len(cols),
                     'MAE':mean_absolute_error(y,pred),'RMSE':mean_squared_error(y,pred)**0.5,
                     'R2':r2_score(y,pred),'bias_pred_minus_actual':float(np.mean(pred-y))})
        oof=b1[['batch','file_name','dilution_percent','zoom']].copy()
        oof['feature_set']=set_name;oof['model']=model_name;oof['prediction']=pred;oof['error']=pred-y
        oof_frames.append(oof)

metrics=pd.DataFrame(rows).sort_values(['MAE','RMSE']).reset_index(drop=True)
metrics.to_csv(out/'v11_v14_b1_grouped_cv_metrics.csv',index=False)
pd.concat(oof_frames,ignore_index=True).to_csv(out/'v11_v14_b1_oof_predictions.csv',index=False)

# Select using B1 only. Refit the winner on all B1 and inspect B2 only as exploratory.
winner=metrics.iloc[0]
set_name=winner.feature_set; model_name=winner.model
train_df,cols=sets[set_name]
train=train_df[train_df.batch=='B1'].copy(); test=train_df[train_df.batch=='B2'].copy()
final_model=models[model_name]
final_model.fit(train[cols].replace([np.inf,-np.inf],np.nan),train.dilution_percent.astype(float))
if len(test):
    bp=final_model.predict(test[cols].replace([np.inf,-np.inf],np.nan))
    b2=pd.DataFrame({'batch':test.batch,'file_name':test.file_name,'actual':test.dilution_percent,'prediction':bp,'error':bp-test.dilution_percent})
    b2.to_csv(out/'v14_b2_exploratory_predictions.csv',index=False)
    b2summary={'n':len(b2),'MAE':float(mean_absolute_error(b2.actual,b2.prediction)),
               'RMSE':float(mean_squared_error(b2.actual,b2.prediction)**0.5),
               'R2':float(r2_score(b2.actual,b2.prediction)),
               'bias_pred_minus_actual':float(b2.error.mean())}
    b2.groupby('actual').agg(n=('error','size'),MAE=('error',lambda x:float(np.mean(np.abs(x)))),
                              mean_prediction=('prediction','mean'),bias=('error','mean')).reset_index().to_csv(out/'v14_b2_by_dilution_exploratory.csv',index=False)
else: b2summary=None

# Fit an ExtraTrees diagnostic on B1 to rank features for the selected feature set.
# Importance is descriptive and not proof of causal relevance.
importance_model=make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,n_jobs=-1,random_state=42))
importance_model.fit(train[cols].replace([np.inf,-np.inf],np.nan),train.dilution_percent.astype(float))
et=importance_model.named_steps['extratreesregressor']
imp=pd.DataFrame({'feature':cols,'importance':et.feature_importances_}).sort_values('importance',ascending=False)
imp.to_csv(out/'v11_feature_importance_selected_set.csv',index=False)

# Feature distribution/batch-shift audit for selected set, on shared feature columns.
shift=[]
for c in cols:
    a=train[c].astype(float).replace([np.inf,-np.inf],np.nan).dropna()
    b=test[c].astype(float).replace([np.inf,-np.inf],np.nan).dropna()
    if len(a) and len(b):
        shift.append({'feature':c,'B1_mean':a.mean(),'B2_mean':b.mean(),'mean_difference_B2_minus_B1':b.mean()-a.mean(),
                      'B1_std':a.std(),'B2_std':b.std()})
pd.DataFrame(shift).to_csv(out/'v11_selected_feature_batch_shift.csv',index=False)

# Dilution x zoom B1 OOF error for the selected model, to expose weak regions.
oof_selected=pd.concat(oof_frames,ignore_index=True)
oof_selected=oof_selected[(oof_selected.feature_set==set_name)&(oof_selected.model==model_name)].copy()
oof_selected['abs_error']=oof_selected.error.abs()
oof_selected.groupby(['dilution_percent','zoom']).agg(n=('abs_error','size'),MAE=('abs_error','mean'),
    mean_prediction=('prediction','mean'),bias=('error','mean')).reset_index().to_csv(out/'v12_b1_dilution_zoom_error.csv',index=False)

summary={
 'inputs':{'unified_csv':str(full_path),'roi_csv':str(roi_path),'unified_shape':list(full.shape),'roi_shape':list(roi.shape),
           'unified_unique_images':int(full.file_name.nunique()),'roi_unique_images':int(roi.file_name.nunique()),
           'roi_sizes':sorted([int(x) for x in roi.roi_size.dropna().unique()]),
           'B1_images':int(full[full.batch=='B1'].file_name.nunique()),'B2_images':int(full[full.batch=='B2'].file_name.nunique())},
 'selection_rule':'Lowest B1 3-fold GroupKFold MAE grouped by dilution-session (dilution_percent + zoom); groups=file_name. B2 not used for model selection.',
 'winner':{'feature_set':str(set_name),'model':str(model_name),'B1_MAE':float(winner.MAE),'B1_RMSE':float(winner.RMSE),'B1_R2':float(winner.R2),'B1_bias':float(winner.bias_pred_minus_actual)},
 'B2_exploratory_only':b2summary,
 'top_15_features':imp.head(15).to_dict('records'),
 'limitations':['B1 grouped CV is internal validation within B1, not independent real-world validation.',
                'B2 is exploratory batch-shift analysis and must not be used to claim independent accuracy.',
                'Feature importance is descriptive; independent fresh samples with reference dilution measurements are still required.',
                'This model estimates water dilution only; it does not establish detection of detergent, soybean or other adulterants.']
}
(out/'v11_v14_summary.json').write_text(json.dumps(summary,indent=2))
print('\nV11-V14 COMBINED ANALYSIS COMPLETE')
print('Output:',out)
print('\nTOP B1 GROUPED-CV MODELS:')
print(metrics.head(12).to_string(index=False))
print('\nSELECTED:',set_name,'+',model_name)
print('B1:',summary['winner'])
print('B2 exploratory:',b2summary)
print('\nTOP FEATURES:')
print(imp.head(15).to_string(index=False))
