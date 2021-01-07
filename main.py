import pathlib 
import numpy as np
from whatlies.language import CountVectorLanguage, UniversalSentenceLanguage, BytePairLanguage, SentenceTFMLanguage
from whatlies.transformers import Pca, Umap
from hulearn.experimental.interactive import InteractiveCharts

txt = pathlib.Path("test_data/nlu.md").read_text()
texts = list(set([t.replace(" - ", "") for t in txt.split("\n") if len(t) > 0 and t[0] != "#"]))
print(str(len(texts)))

lang_cv = CountVectorLanguage(10)
lang_use = UniversalSentenceLanguage("large")
lang_bp = BytePairLanguage("en",dim=300,vs=200_00)

def make_plot(lang):
    return(lang[texts]
           .transform(Umap(2))
           .plot_interactive(annot=False)
           .properties(width=250, height=250, title=type(lang).__name__))
make_plot(lang_cv) | make_plot(lang_bp) | make_plot(lang_use)

print('Eclipse Che rocks!')
