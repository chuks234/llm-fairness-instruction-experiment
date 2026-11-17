"""BBQ two-model instruction-condition comparison. Run: streamlit run bbq_comparison_dashboard.py"""
import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import binomtest

st.set_page_config(page_title='BBQ | GPT-6-Luna vs Qwen3', layout='wide')
CATS=['Age','Disability_status','Gender_identity','Nationality','Physical_appearance']
ROOT=Path('/content/BBQ_Experiment')
LOCAL_ZIP=Path('/content/BBQ_matched_source_and_GPT6_results.zip')
QWEN=Path('/content/BBQ_Two_Model_Results/qwen3_responses.csv')
MODEL_NAMES={'gpt':'GPT-6-Luna','qwen':'Qwen3-4B-Instruct-2507'}
REQUIRED=['example_id','category','condition','llm_response','correct_answer_index','context_condition']

def read_csv(f):
    return pd.read_csv(f, dtype={'example_id':str,'llm_response':str})

def source_files(qwen_upload,zip_upload):
    if zip_upload is not None:
        with zipfile.ZipFile(io.BytesIO(zip_upload.getvalue())) as z:
            files={Path(n).name:z.read(n) for n in z.namelist() if n.endswith('.csv')}
        g=[]
        for cat in CATS:
            for cond in ['baseline','current']:
                name=f'{cat}_{cond}_results.csv'
                if name not in files: raise ValueError(f'Missing GPT CSV in ZIP: {name}')
                g.append(read_csv(io.BytesIO(files[name])))
        original=pd.concat(g,ignore_index=True)
        sources={cat:read_csv(io.BytesIO(files[f'{cat}.csv'])) for cat in CATS if f'{cat}.csv' in files}
    elif LOCAL_ZIP.exists():
        with zipfile.ZipFile(LOCAL_ZIP) as z:
            g=[read_csv(io.BytesIO(z.read(f'gpt6_results/{cat}_{cond}_results.csv'))) for cat in CATS for cond in ['baseline','current']]
            sources={cat:read_csv(io.BytesIO(z.read(f'processed/{cat}.csv'))) for cat in CATS}
        original=pd.concat(g,ignore_index=True)
    else:
        g=[]
        for cat in CATS:
            for cond in ['baseline','current']:
                p=ROOT/'gpt6_results'/f'{cat}_{cond}_results.csv'
                if not p.exists(): raise FileNotFoundError(f'Cannot find {p}; upload the source ZIP in sidebar.')
                g.append(read_csv(p))
        original=pd.concat(g,ignore_index=True)
        sources={cat:read_csv(ROOT/'processed'/f'{cat}.csv') for cat in CATS if (ROOT/'processed'/f'{cat}.csv').exists()}
    if qwen_upload is not None:
        qwen=read_csv(qwen_upload)
    elif QWEN.exists():
        qwen=read_csv(QWEN)
    else:
        raise FileNotFoundError('Upload qwen3_responses.csv in the sidebar.')
    return original,qwen,sources

def prepare(gpt,qwen,sources):
    parts=[]
    for frame,model in [(gpt,MODEL_NAMES['gpt']),(qwen,MODEL_NAMES['qwen'])]:
        d=frame.copy()
        missing=set(REQUIRED)-set(d.columns)
        if missing: raise ValueError(f'{model} missing columns: {sorted(missing)}')
        d['model']=model
        d['example_id']=d['example_id'].astype(str).str.strip()
        d['category']=d['category'].astype(str).str.strip()
        d['condition']=d['condition'].astype(str).str.strip()
        d['context_condition']=d['context_condition'].astype(str).str.strip().str.lower()
        d['answer']=d['llm_response'].fillna('').astype(str).str.strip()
        d['gold']=pd.to_numeric(d['correct_answer_index'],errors='coerce')
        d['valid']=d['answer'].isin(['0','1','2'])
        d['correct']=d['valid'] & (pd.to_numeric(d['answer'],errors='coerce')==d['gold'])
        d['unknown']=False  # replaced using the matched ambiguous source gold answer below
        d['key']=d['category']+'|'+d['example_id']
        if d.duplicated(['key','condition']).any(): raise ValueError(f'{model} contains duplicate question-condition rows.')
        parts.append(d)
    all_df=pd.concat(parts,ignore_index=True)
    # BBQ answer options are not always ordered with UNKNOWN at index 2.
    # For each matched ambig/disambig pair, the ambiguous gold answer is UNKNOWN.
    source_rows=[]
    for cat,src in sources.items():
        t=src.head(50)[['example_id','correct_answer_index','context_condition']].copy()
        t['example_id']=t.example_id.astype(str).str.strip()
        t['pair_id']=pd.to_numeric(t.example_id,errors='raise').astype(int)//2
        amb=t[t.context_condition.astype(str).str.lower()=='ambig']
        for _,r in amb.iterrows():
            source_rows.append({'category':cat,'pair_id':r.pair_id,'unknown_index':int(r.correct_answer_index)})
    if len(source_rows)!=125:
        raise ValueError(f'Cannot verify UNKNOWN answer positions: expected 125 matched pairs, found {len(source_rows)}.')
    unknown_lookup=pd.DataFrame(source_rows)
    all_df['pair_id']=pd.to_numeric(all_df.example_id,errors='raise').astype(int)//2
    all_df=all_df.merge(unknown_lookup,on=['category','pair_id'],how='left',validate='many_to_one')
    if all_df.unknown_index.isna().any():
        raise ValueError('UNKNOWN option mapping incomplete.')
    all_df['unknown']=all_df.valid & (pd.to_numeric(all_df.answer,errors='coerce')==all_df.unknown_index)
    all_df['target_selected']=all_df.valid & (pd.to_numeric(all_df.answer,errors='coerce')==pd.to_numeric(all_df.target_loc,errors='coerce'))
    for model,d in all_df.groupby('model'):
        if len(d)!=500: raise ValueError(f'{model}: expected 500 rows, found {len(d)}')
        for cat in CATS:
            for cond in ['baseline','current']:
                n=len(d[(d.category==cat)&(d.condition==cond)])
                if n!=50: raise ValueError(f'{model}, {cat}, {cond}: expected 50, found {n}')
        b=set(d[d.condition=='baseline'].key)
        c=set(d[d.condition=='current'].key)
        if b!=c: raise ValueError(f'{model}: baseline/current question IDs do not match')
    sets=[set(d.key) for _,d in all_df.groupby('model')]
    if sets[0]!=sets[1]: raise ValueError('GPT and Qwen question IDs do not match.')
    gold_check=all_df.groupby(['key','condition'])[['gold','context_condition']].nunique()
    if (gold_check>1).any().any(): raise ValueError('Gold answers or context types differ across models.')
    if sources:
        for cat,src in sources.items():
            expected=set(src.head(50).example_id.astype(str))
            observed=set(all_df[all_df.category==cat].example_id)
            if expected!=observed: raise ValueError(f'{cat}: results do not match first 50 original source questions')
    return all_df

def summaries(df):
    keys=['model','category','context_condition','condition']
    detail=df.groupby(keys,as_index=False).agg(n=('correct','size'),correct_n=('correct','sum'),valid_n=('valid','sum'),unknown_n=('unknown','sum'))
    detail['accuracy_pct']=100*detail.correct_n/detail.n
    detail['unknown_pct']=100*detail.unknown_n/detail.n
    return detail

def paired(df):
    p=df.pivot(index=['model','key','category','context_condition'],columns='condition',values=['correct','answer','unknown'])
    if p.isna().any().any(): raise ValueError('Incomplete baseline/current pairing')
    p.columns=[f'{a}_{b}' for a,b in p.columns]
    p=p.reset_index()
    for x in ['correct','unknown']:
        for cond in ['baseline','current']:
            p[f'{x}_{cond}']=p[f'{x}_{cond}'].astype(bool)
    p['changed']=p.answer_baseline!=p.answer_current
    p['correct_to_wrong']=p.correct_baseline & ~p.correct_current
    p['wrong_to_correct']=~p.correct_baseline & p.correct_current
    return p

def stats(p,seed=42,iterations=5000):
    rng=np.random.default_rng(seed)
    records=[]
    for (model,category,context),g in p.groupby(['model','category','context_condition'],dropna=False):
        x=g.correct_baseline.to_numpy(dtype=float)
        y=g.correct_current.to_numpy(dtype=float)
        delta=(y-x)*100
        n=len(g)
        indices=rng.integers(0,n,size=(iterations,n))
        boot=delta[indices].mean(axis=1)
        cw=int(g.correct_to_wrong.sum());wc=int(g.wrong_to_correct.sum())
        discord=cw+wc
        pv=float(binomtest(min(cw,wc),discord,0.5).pvalue) if discord else 1.0
        records.append({'model':model,'category':category,'context_condition':context,'n_pairs':n,
            'baseline_accuracy_pct':100*x.mean(),'current_accuracy_pct':100*y.mean(),
            'delta_pp':delta.mean(),'ci_low_pp':np.percentile(boot,2.5),'ci_high_pp':np.percentile(boot,97.5),
            'changed_n':int(g.changed.sum()),'correct_to_wrong':cw,'wrong_to_correct':wc,'mcnemar_exact_p':pv,
            'baseline_unknown_pct':100*g.unknown_baseline.mean(),'current_unknown_pct':100*g.unknown_current.mean()})
    return pd.DataFrame(records)

def overall_stats(p,iterations=5000):
    rng=np.random.default_rng(777)
    out=[]
    for model,g in p.groupby('model'):
        for label,sub in [('All',g),('Ambiguous',g[g.context_condition=='ambig']),('Disambiguated',g[g.context_condition=='disambig'])]:
            if sub.empty: continue
            x=sub.correct_baseline.to_numpy(dtype=float); y=sub.correct_current.to_numpy(dtype=float)
            diff=(y-x)*100;n=len(diff)
            ix=rng.integers(0,n,size=(iterations,n));bs=diff[ix].mean(axis=1)
            cw=int(sub.correct_to_wrong.sum());wc=int(sub.wrong_to_correct.sum())
            out.append({'model':model,'context':label,'n_pairs':n,'baseline_accuracy_pct':100*x.mean(),
                        'current_accuracy_pct':100*y.mean(),'delta_pp':diff.mean(),
                        'ci_low_pp':np.percentile(bs,2.5),'ci_high_pp':np.percentile(bs,97.5),
                        'changed_n':int(sub.changed.sum()),'correct_to_wrong':cw,'wrong_to_correct':wc,
                        'mcnemar_exact_p':binomtest(min(cw,wc),cw+wc).pvalue if cw+wc else 1.0,
                        'baseline_unknown_pct':100*sub.unknown_baseline.mean(),
                        'current_unknown_pct':100*sub.unknown_current.mean()})
    return pd.DataFrame(out)

def did_stats(p,iterations=5000):
    rng=np.random.default_rng(191)
    out=[]
    for context,sub in [('All',p),('Ambiguous',p[p.context_condition=='ambig']),('Disambiguated',p[p.context_condition=='disambig'])]:
        wide=sub.pivot(index='key',columns='model',values=['correct_baseline','correct_current'])
        wide=wide.dropna()
        if wide.empty: continue
        q=(wide[('correct_current',MODEL_NAMES['qwen'])].astype(float)-wide[('correct_baseline',MODEL_NAMES['qwen'])].astype(float)).to_numpy()*100
        g=(wide[('correct_current',MODEL_NAMES['gpt'])].astype(float)-wide[('correct_baseline',MODEL_NAMES['gpt'])].astype(float)).to_numpy()*100
        diff=q-g;n=len(diff);ix=rng.integers(0,n,size=(iterations,n));bs=diff[ix].mean(axis=1)
        out.append({'context':context,'n_matched_questions':n,'qwen_delta_pp':q.mean(),'gpt_delta_pp':g.mean(),
                    'difference_in_differences_pp':diff.mean(),'ci_low_pp':np.percentile(bs,2.5),'ci_high_pp':np.percentile(bs,97.5)})
    return pd.DataFrame(out)

def png(fig):
    try: return fig.to_image(format='png',scale=2)
    except Exception: return None

def export_zip(tables,figures):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for name,frame in tables.items():z.writestr(f'tables/{name}.csv',frame.to_csv(index=False))
        for name,fig in figures.items():
            data=png(fig)
            if data:z.writestr(f'figures/{name}.png',data)
            z.writestr(f'figures/{name}.html',fig.to_html(full_html=True,include_plotlyjs='cdn'))
        z.writestr('README.txt','BBQ two-model analysis. Delta = current minus baseline in percentage points.\n'
                   '95% CIs: paired bootstrap, 5000 resamples; difference-in-differences resamples matched questions.\n'
                   'McNemar p-values: exact two-sided binomial test of discordant pairs.\n'
                   'UNKNOWN option mapped using each matched ambiguous source question.\n'
                   'This dashboard does not compute official BBQ stereotype-bias scores.\n')
    return b.getvalue()

st.title('BBQ fairness evaluation | Two-model comparison')
st.caption('GPT-6-Luna vs Qwen3-4B-Instruct-2507 · Baseline vs modified instructions · Matched BBQ questions')
with st.sidebar:
    st.header('Input files')
    st.write('In the original Colab runtime, the dashboard reads the saved files automatically. Otherwise upload both files below.')
    qwen_file=st.file_uploader('Qwen3 responses CSV',type=['csv'])
    source_zip=st.file_uploader('Original BBQ + GPT-6 results ZIP',type=['zip'])
try:
    gpt,qwen,sources=source_files(qwen_file,source_zip)
    all_df=prepare(gpt,qwen,sources)
except Exception as exc:
    st.error(f'Data loading/validation: {exc}')
    st.info('If running locally, upload qwen3_responses.csv and BBQ_matched_source_and_GPT6_results.zip in the sidebar.')
    st.stop()

p=paired(all_df)
summary=summaries(all_df)
stat=stats(p)
overall=overall_stats(p)
did=did_stats(p)
with st.sidebar:
    st.header('Filters')
    categories=st.multiselect('Demographic categories',CATS,default=CATS)
    contexts=st.multiselect('Context types',sorted(all_df.context_condition.unique()),default=sorted(all_df.context_condition.unique()))
    models=st.multiselect('Models',list(MODEL_NAMES.values()),default=list(MODEL_NAMES.values()))
filtered=all_df[all_df.category.isin(categories)&all_df.context_condition.isin(contexts)&all_df.model.isin(models)]
fp=p[p.category.isin(categories)&p.context_condition.isin(contexts)&p.model.isin(models)]
if filtered.empty:
    st.warning('No records match the filters.');st.stop()

m1,m2,m3,m4=st.columns(4)
m1.metric('Responses displayed',f'{len(filtered):,}')
m2.metric('Matched questions',f'{filtered.key.nunique():,}')
m3.metric('Valid answers',f'{filtered.valid.mean()*100:.1f}%')
m4.metric('Answer changes',f'{fp.changed.sum():,}')

figures={}
tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs(['Overview','Demographics','Context & UNKNOWN','Response changes','Statistical tests','Downloads','Tables & heatmap'])
with tab1:
    st.subheader('Overall accuracy')
    ov=(filtered.groupby(['model','condition'],as_index=False).agg(n=('correct','size'),accuracy=('correct','mean')))
    ov['accuracy_pct']=ov.accuracy*100
    fig=px.bar(ov,x='model',y='accuracy_pct',color='condition',barmode='group',text='accuracy_pct',range_y=[0,100],labels={'accuracy_pct':'Accuracy (%)','model':'Model'})
    fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
    st.plotly_chart(fig,use_container_width=True);figures['overall_accuracy']=fig
    st.dataframe(ov[['model','condition','n','accuracy_pct']],hide_index=True,use_container_width=True)
    st.subheader('Full-experiment paired changes (unfiltered)')
    st.dataframe(overall.round(3),hide_index=True,use_container_width=True)
with tab2:
    st.subheader('Accuracy by demographic category')
    cats=(filtered.groupby(['model','category','condition'],as_index=False).correct.mean())
    cats['accuracy_pct']=100*cats.correct
    fig=px.bar(cats,x='category',y='accuracy_pct',color='condition',facet_row='model',barmode='group',range_y=[0,100],labels={'accuracy_pct':'Accuracy (%)','category':'BBQ category'})
    fig.update_layout(height=650)
    st.plotly_chart(fig,use_container_width=True);figures['category_accuracy']=fig
    st.dataframe(cats[['model','category','condition','accuracy_pct']].round(2),hide_index=True,use_container_width=True)
    delta=(fp.groupby(['model','category'],as_index=False).apply(lambda g:pd.Series({'delta_pp':100*(g.correct_current.astype(int)-g.correct_baseline.astype(int)).mean()}),include_groups=False))
    if not delta.empty:
        hm=delta.pivot(index='model',columns='category',values='delta_pp')
        fig=px.imshow(hm,text_auto='.1f',color_continuous_scale='RdBu',color_continuous_midpoint=0,labels={'color':'Current − baseline (pp)'},aspect='auto')
        st.plotly_chart(fig,use_container_width=True);figures['demographic_drift_heatmap']=fig
with tab3:
    st.subheader('Ambiguous vs disambiguated accuracy')
    context=(filtered.groupby(['model','context_condition','condition'],as_index=False).agg(accuracy=('correct','mean'),unknown=('unknown','mean')))
    context['accuracy_pct']=100*context.accuracy;context['unknown_pct']=100*context.unknown
    fig=px.bar(context,x='context_condition',y='accuracy_pct',color='condition',facet_col='model',barmode='group',range_y=[0,100])
    st.plotly_chart(fig,use_container_width=True);figures['context_accuracy']=fig
    st.subheader('Verified UNKNOWN response rate')
    fig=px.bar(context,x='context_condition',y='unknown_pct',color='condition',facet_col='model',barmode='group',range_y=[0,100])
    st.plotly_chart(fig,use_container_width=True);figures['unknown_rate']=fig
    st.caption('UNKNOWN is mapped from the correct answer in each matched ambiguous BBQ source item, not assumed to be option 2.')
    st.dataframe(context[['model','context_condition','condition','accuracy_pct','unknown_pct']].round(2),hide_index=True,use_container_width=True)
with tab4:
    st.subheader('Within-model response instability')
    changes=(fp.groupby(['model','category'],as_index=False).agg(changed_n=('changed','sum'),correct_to_wrong=('correct_to_wrong','sum'),wrong_to_correct=('wrong_to_correct','sum')))
    fig=px.bar(changes,x='category',y='changed_n',color='model',barmode='group')
    st.plotly_chart(fig,use_container_width=True);figures['response_changes']=fig
    st.dataframe(changes,hide_index=True,use_container_width=True)
    st.subheader('Individual changed answers')
    st.dataframe(fp[fp.changed][['model','key','context_condition','answer_baseline','answer_current','correct_to_wrong','wrong_to_correct']],hide_index=True,use_container_width=True)
with tab5:
    st.subheader('Exact McNemar tests and paired bootstrap 95% confidence intervals')
    st.caption('Exact McNemar tests are within-model, paired baseline/current tests. Bootstrap confidence intervals use 5,000 paired resamples; p-values are unadjusted for multiple comparisons.')
    st.dataframe(overall.round(4),hide_index=True,use_container_width=True)
    st.subheader('By category and context')
    st.dataframe(stat[stat.model.isin(models)&stat.category.isin(categories)&stat.context_condition.isin(contexts)].round(4),hide_index=True,use_container_width=True)
    st.subheader('Cross-model difference-in-differences')
    st.caption('Qwen instruction effect minus GPT instruction effect, computed on matched question IDs; 95% CIs resample questions jointly across models.')
    st.dataframe(did.round(4),hide_index=True,use_container_width=True)
    # Error bar lengths must be distances, not absolute endpoints
    plot=overall[overall.model.isin(models)].copy()
    plot['plus']=plot.ci_high_pp-plot.delta_pp;plot['minus']=plot.delta_pp-plot.ci_low_pp
    fig=px.scatter(plot,x='delta_pp',y='model',color='context',error_x='plus',error_x_minus='minus',labels={'delta_pp':'Accuracy change (pp)'})
    fig.add_vline(x=0,line_dash='dash')
    st.plotly_chart(fig,use_container_width=True);figures['paired_confidence_intervals']=fig
with tab6:
    st.subheader('Download publication tables and charts')
    tables={'all_responses':all_df,'paired_responses':p,'category_context_summary':summary,'category_context_statistics':stat,'overall_statistics':overall,'cross_model_difference_in_differences':did}
    st.download_button('Download all tables + figures (ZIP)',export_zip(tables,figures),file_name='BBQ_Two_Model_Analysis.zip',mime='application/zip',type='primary')
    st.download_button('Download combined 1,000-response dataset (CSV)',all_df.to_csv(index=False),file_name='BBQ_combined_responses.csv',mime='text/csv')
    st.caption('The ZIP includes all tables, and charts generated in visited tabs. PNG exports require Kaleido; interactive HTML charts are always included.')


with tab7:
    st.subheader('Table A1 · Two-model comparison')
    st.caption('Baseline and modified instruction results; Δ is modified minus baseline in percentage points. Filters apply to the selected models, categories, and contexts.')
    metric_rows=[]
    for (model,condition),g in filtered.groupby(['model','condition']):
        metric_rows.append({'Model':model,'Condition':condition,'N':len(g),
                            'Accuracy (%)':100*g.correct.mean(),
                            'Target selection (%)':100*g.target_selected.mean(),
                            'UNKNOWN response (%)':100*g.unknown.mean()})
    metric_table=pd.DataFrame(metric_rows)
    instability=(fp.groupby('model').changed.mean()*100).to_dict()
    metric_table['Response instability (%)']=metric_table.Model.map(instability)
    st.dataframe(metric_table.round(2),hide_index=True,use_container_width=True)
    st.caption('Response instability is a paired model-level measure, repeated for baseline and modified rows. Target selection is not equivalent to a validated stereotype-direction bias score; DBS is deliberately not reported.')
    if not metric_table.empty:
        pivot=metric_table.pivot(index='Model',columns='Condition',values=['Accuracy (%)','Target selection (%)','UNKNOWN response (%)'])
        if {'baseline','current'}.issubset(metric_table.Condition.unique()):
            delta_table=pd.DataFrame({'Model':pivot.index})
            for metric in ['Accuracy (%)','Target selection (%)','UNKNOWN response (%)']:
                delta_table[metric.replace(' (%)',' Δ (pp)')]=(pivot[(metric,'current')]-pivot[(metric,'baseline')]).values
            st.subheader('Change under modified instructions')
            st.dataframe(delta_table.round(2),hide_index=True,use_container_width=True)
    st.subheader('Polarity-stratified comparison')
    polarity=(filtered.groupby(['model','question_polarity','condition'],as_index=False)
              .agg(N=('correct','size'),accuracy_pct=('correct',lambda x:100*x.mean()),
                   target_pct=('target_selected',lambda x:100*x.mean()),
                   unknown_pct=('unknown',lambda x:100*x.mean())))
    st.dataframe(polarity.rename(columns={'model':'Model','question_polarity':'Polarity','condition':'Condition',
              'accuracy_pct':'Accuracy (%)','target_pct':'Target selection (%)','unknown_pct':'UNKNOWN (%)'}).round(2),hide_index=True,use_container_width=True)
    st.subheader('Heatmap · Accuracy change by category and model')
    heat=fp.groupby(['model','category']).apply(
        lambda g:100*(g.correct_current.astype(int)-g.correct_baseline.astype(int)).mean(),
        include_groups=False).unstack('category')
    if not heat.empty:
        heat=heat.reindex(columns=[c for c in CATS if c in heat.columns])
        fig=px.imshow(heat,text_auto='.1f',color_continuous_scale='RdBu',color_continuous_midpoint=0,
                      aspect='auto',labels={'color':'Accuracy Δ (pp)','x':'BBQ category','y':'Model'})
        fig.update_layout(height=350)
        st.plotly_chart(fig,use_container_width=True)
        figures['publication_accuracy_drift_heatmap']=fig
    st.download_button('Download Table A1 (CSV)',metric_table.to_csv(index=False),file_name='BBQ_Table_A1.csv',mime='text/csv')
    st.download_button('Download polarity table (CSV)',polarity.to_csv(index=False),file_name='BBQ_Polarity_Table.csv',mime='text/csv')

st.divider()
st.caption('Interpretation: accuracy drift is not the same as stereotype bias. Results are observational comparisons of the specified instruction conditions; do not infer model-wide fairness properties from these five-category samples.')
