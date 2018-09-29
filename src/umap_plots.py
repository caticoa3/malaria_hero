# coding: utf-8

import numpy as np
import pandas as pd
import umap
from bokeh.resources import INLINE, CDN
from bokeh.embed import file_html
#https://umap-learn.readthedocs.io/en/latest/basic_usage.html

def embeddable_image(image_path):
    from io import BytesIO
    from PIL import Image
    import base64
    
    image = Image.open(str(image_path)).resize((64, 64), Image.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()


def umap_bokeh(bn_feat, #= pd.read_csv('../results/prod_test_feat.csv', index_col=0), 
               pred_df, #= pd.read_csv('../results/predicted_malaria.csv', index_col = 0),
               image_folder = '../flask/uploads'
               ):
#feat_file = '../results/prod_test_feat.csv'
#prediction_csv = '../results/predictions_malaria.csv'
#image_folder = '../flask/uploads'

#bn_feat = pd.read_csv('../data/cv_feat.csv', index_col=0)
#pred_df = pd.read_csv('../results/predictions_prod_test.csv', index_col = 0)
#image_folder = '../flask/uploads'

#load 

    '''This function is for plotting a bokeh plot of UMAP dimensionality 
    reduction plot, colored based on labels, with thumnail overlays for
    webapp results.'''
    if bn_feat.shape[0] < 3:
        print('Please select more than 3 cells to classify')
        return

#    bn_feat = pd.read_csv(feat_file, index_col = 0)
    #bn_feat = bn_feat.sample(frac=0.01)
    reducer = umap.UMAP(random_state=42)
    
    ## -- UMAP (might want to make into another function)
    #Train the dimonsionality reduction
    features = bn_feat.drop(columns=['label','fn'])
    
    reducer.fit(features)

    embedding = reducer.transform(features)
    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert(np.all(embedding == reducer.embedding_))
    embedding.shape
    
    #    mask = df.label == 'Parasitized'
    
    bn_feat['path'] = str(image_folder) +'/' + bn_feat['fn']
     
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
    from bokeh.palettes import Spectral10
    from bokeh.embed import components
    output_notebook()
    
    df_images = pd.DataFrame(embedding, columns=('x','y'), index=bn_feat.index)
    df_images['image'] = list(map(embeddable_image, list(bn_feat['path'].values)))
    df_images['label'] = pred_df.loc[:,['Predicted_label']].astype(str)
    
    datasource = ColumnDataSource(df_images)
    color_mapping = CategoricalColorMapper(factors = [str(x) for x in list(set(df_images.label.astype(str).values))],
                                           palette=Spectral10)
    
    plot_figure = figure(
        title='UMAP projection of the malaria dataset',
        plot_width=600,
        plot_height=600,
        tools=('pan, wheel_zoom, reset')
    )
    
    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 12px; color: #224499'>Class:</span>
            <br /><span style='font-size: 12px'>@label</span>
        </div>
    </div>
    """))
    
    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='label', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )
    #    show(plot_figure)
    script, div = components(plot_figure)
#    html = file_html(plot_figure, INLINE)

    return script, div #html

