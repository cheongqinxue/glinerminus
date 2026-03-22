# GlinerMinus

GLiNER is designed as a NER model that is able to identify any type of entity. Inspired by GLiNER, 
this repository seeks to build a model that only verifies if a span of text is a match for one or 
more entity types.

We use a vastly simplified architecture. Similar to GLiNER, entity types are encoded into the same 
latent semantic space as the candidate spans. Entity type prompts are inserted before the input 
sentence and a special \<ent\> token is placed before each one. The entire sequence is then sent
to the backbone model to extract token level features.

Unlike GLiNER, we do not employ the use of additional feedforward layers to refine the features.
Instead, we take the emitted features at each of the \<ent\> tokens and perform a bilinear operation
with the entity features and the candidate spans' features as inputs. The bilinear operation 
provides as output 3 logits, corresponding with 0: "Match", 1: "Neutral", 2: "No Match". The 
addition of the Neutral class allows the model to be trained to abstain when facing out-of-domain
data.

## Credits

This repository was inspired by [GLiNER](https://github.com/urchade/GLiNER) (Generalist and Lightweight Model for Named Entity Recognition). If you use this work, please consider citing the original GLiNER paper:

```bibtex
@inproceedings{zaratiana-etal-2024-gliner,
    title = "{GL}i{NER}: Generalist Model for Named Entity Recognition using Bidirectional Transformer",
    author = "Zaratiana, Urchade and Tomeh, Nadi and Holat, Pierre and Charnois, Thierry",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.300",
    doi = "10.18653/v1/2024.naacl-long.300",
    pages = "5364--5376",
}
```

Paper: [ACL Anthology](https://aclanthology.org/2024.naacl-long.300)
