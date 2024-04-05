# LLMs_withTransformers


## <strong>Introduction:</strong> this LLM model is one the varients of GPT models but relatively smaller one (NanoGPT), a single book is used to train this model.

## tools used are `Python`, `PyTorch`, and `Matplotlib`

## The data used to train is a single book as mentioned, [The wizerd of Oz](https://en.wikipedia.org/wiki/The_Wizard_of_Oz), <br> dictionary: `9333 words` <br> length: `38261 words`

## for the arch. part, <strong>[transformers](https://miro.medium.com/v2/resize:fit:800/1*GIVM8Wat6Vq8W7Eff-f_5w.png)</strong> is used, to get help of its multihead attention, <br> the hyper-parameters: <br> batch_size = `64` <br> block_size = `256` <br> max_iters = `5000` <br> eval_interval = `500` <br> learning_rate = `3e-4` <br> eval_iters = `200` <br> n_embd = `384` <br> n_head = `6` <br> n_layer = `6` <br> dropout = `0.2`

## <strong>Conclusion: </strong> In this project, we developed a Language Model (LM) based on the NanoGPT architecture, trained using text from "The Wizard of Oz." Leveraging Python, PyTorch, and the Transformers library, we optimized hyperparameters for effective training. The model demonstrated the ability to generate coherent text, showcasing the effectiveness of transformer-based LMs in natural language processing tasks, even with limited data and computational resources.

<br>

## <strong> References: </strong>
### - "The Wizard of Oz." Wikipedia, Wikimedia Foundation, 25 Mar. 2024. <br> - Vaswani, Ashish, et al. "Attention Is All You Need." Advances in Neural Information Processing Systems, vol. 30, 2017. <br> - Wolf, Thomas, et al. "Hugging Face's Transformers: State-of-the-art Natural Language Processing." ArXiv:2003.05565 [Cs], 2020. <br> - PyTorch Documentation. <br> - Hugging Face Transformers Documentation.