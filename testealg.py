import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors

movies = pd.read_csv('movies.csv', sep=',')
ratings = pd.read_csv('ratings.csv', sep=',')

#print(movies.shape)
#print(ratings.head())


'''
Nesse primeiro passo a gente apenas lê o arquivo csv e o transforma em dataframe.
Dataframe é uma estrutura de dados do Pandas. Organiza os dados de forma tabulada - uma tabela.
Existem outros métodos de read. No nosso caso, nosso dataset ta em formato csvThis, então pegamos o método .read_csv
ler documentação para entender outros parâmetros.
https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
'''
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')



'''
Aqui nós removemos linhas duplicadas.
O método .drop_duplicates remove linhas duplicadas tanto de items como de séries.
Um argumento interessante é o 'subset= ', esse parâmetro remove linhas duplicadas só com base em certas colunas.
O argumento inplace diz se o método vai alterar o dataframe original (inplace=true) ou retornar um novo(inplace=false)
'''
movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)


'''
Aqui nós removemos linhas com valores faltantes.
O método .drop_duplicates remove linhas - ou colunas - com valores faltantes (NaN, not a number)
O argumento inplace diz se o método vai alterar o dataframe original (inplace=true) ou retornar um novo(inplace=false)
Existem argumentos que definem as regras de retirada - se é de acordo com linha, coluna etc
'''
movies.dropna(inplace=True)
ratings.dropna(inplace=True)


'''
Aqui pegamos a coluna do nosso dataframe, representado como vetor unidimensional
Quando fazemos isso, estamos pegando o que chamamos de series.
O Pandas percorre cada célula da coluna generos, aplica o método .split('|') em cada string 
substitui a string original pela nova lista de strings que foi criada. 
'''
genres = movies['genres'].str.split('|')


'''
Aqui criamos nossa matriz item x característica
As colunas serão nossas features e as linhas os items
'''
encoderMB = MultiLabelBinarizer(sparse_output = False)
genres_encoded = encoderMB.fit_transform(genres)


'''
Aqui estamos ensinando nosso modelo a achar os vizinhos mais proximos de um vetor filme
A métrica que usamos é o cosine
'''
recommender = NearestNeighbors(metric='cosine')
recommender.fit(genres_encoded)


'''
aqui estamos escolhendo qual o usuário a quem queremos recomendar um filme
você pode alterar e escolher o usuário que quiser, consulte o csv ratings
'''
dfUser = ratings[ratings['userId'] == 611] 
#print(dfUser.shape)
#print(dfUser.head())


'''
Criando a base do vetor que representará o usuário
'''
vetor_usuario = np.zeros(genres_encoded.shape[1])
#print(vetor_usuario)


'''
Preecnhendo o vetor do usuário com suas informações, onde cada indice tera a avaliacao do seu respectivo filme
'''
for row in dfUser.itertuples():
    movieId = row[2]
    indexMovie = movies[movies['movieId'] == movieId].index[0]
    new_vec = genres_encoded[indexMovie] * row[3]
    vetor_usuario += new_vec
    #print(vetor_usuario)
    #print("\n")

vetor_usuario = vetor_usuario/genres.shape[0]
#print(vetor_usuario)


'''
Aqui, usamos o perfil de gosto do usuário que acabamos de calcular para encontrar os filmes mais recomendados. 
O modelo NearestNeighbors busca no espaço de características os filmes em que vetores são próximos do vetor do nosso usuário. 
O método .kneighbors realiza essa busca, retornando os 15 vizinhos mais próximos (n_neighbors=15). 
O resultado é uma lista de índices, que usamos para achar e exibir os nomes do filme no dataframe.
'''
recommender = NearestNeighbors()
recommender.fit(genres_encoded)
vizinhos = recommender.kneighbors(vetor_usuario.reshape(1,-1),n_neighbors=15, return_distance=False)
titulosVizinhos = movies.iloc[vizinhos[0]]['title']

print(titulosVizinhos)



