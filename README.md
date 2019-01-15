# Email_summarization
The Enron Email dataset contains 500,000+ emails from enron employees. It can be downloaded from Kaggle(https://www.kaggle.com/wcukierski/enron-email-dataset).

The module uses code of the Skip-Thoughts paper which can be found here:
git clone https://github.com/ryankiros/skip-thoughts

Download the pre-trained models. The total download size will be of around 5 GB. Do:
mkdir skip-thoughts/models
wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
