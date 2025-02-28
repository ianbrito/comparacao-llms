import ollama
import spacy
import math
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rouge_score import rouge_scorer
from bert_score import BERTScorer

texto_original = "A Inteligência Artificial é um tema que está em voga. Reiteradamente ouvimos aspessoas falando acerca de profissionais da área de tecnologia ao meio acadêmico e a sociedadeem geral. A Inteligência Artificial (IA) é uma subárea dentro da Ciência da Computaçãoresponsável por pesquisar e propor a elaboração de dispositivos computacionais que tentamsimular o intelecto humano (Silva, 2013). No contexto das organizações, podem ajudar namelhoria da efetividade e produtividade (Ransbotham et al, 2021). No entanto, como campointerdisciplinar, a IA possui o seu tentáculo em múltiplas áreas do conhecimento, tais comomatemática, robótica e ciência da computação (Russell; Norvig, 2013). Neste sentido, esta áreaé de fundamental relevância, ao passo que facilita no desenvolvimento de trabalho de formaeficiente e eficaz, principalmente nas áreas de conhecimento que hoje estão integradas com aIA. No entanto, Sousa (2023), alerta para algumas desvantagens que a utilização inadequada daIA pode trazer nos campos de ensino em EAD, tais como: alta dependência das infraestruturase dependência digital em excesso.Diante deste contexto, percebe-se o aumento da utilização da IA em pesquisasacadêmicas e profissionais (Gontijo; Araújo, 2021). Assim, o objetivo do trabalho foi estudar aelaboração de resumos feitos com ferramentas de IA, no intuito de verificar as diferençassubstantivas entre os resumos, utilizando como fonte quatro artigos na área da Ética. Além dedesenvolver a capacidade analítica no desenvolvimento dos resumos, foram feitas reflexõesacerca da capacidade da IA. A seguir, apresentamos a tabela resumo das aplicações.Todos os resumos gerados pela IA tinham qualidade suficiente para o uso na áreaacadêmica, sendo que os gerados pelo Humata foram os que apresentaram melhor consistênciae coesão. A ferramenta Humata oferece apenas um resumo, com os parágrafos contínuos. Notocante aos resumos gerados pelo Resoomer, embora coerentes, foram apresentados porparágrafos e não em texto continuo, em formato de tópicos para a cada parágrafo ou empassagens mais relevantes. Além disso, eram mais longos e, na maioria das vezes, com duaslaudas em média. Com o Tome, os resumos gerados foram coerentes, mas também são geradosem tópicos e em formato de slides. Por final, foi possível verificar e comparar os resumosefetuados pelos bolsistas de iniciação científica. Enquanto dois textos tiveram como ponto fortea coesão, outros dois foram destaque na coerência. Com este exercício de comparação, foipossível perceber que os resumos gerados por IA e os bolsistas de iniciação cientificacumpriram com os requisitos. No entanto, apesar das potencialidades e vantagens com o uso deferramentas de IA para assistência da escrita acadêmica, é importante ressaltar os limites. Comoconclusão, distingue-se que a sabedoria está em utilizar de forma prudente as potencialidades edirimir os limites."


def rouge(texto_original, texto_gerado):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(texto_original, texto_gerado)

    return {
        "rouge-1": scores["rouge1"].fmeasure,
        "rouge-2": scores["rouge2"].fmeasure,
        "rouge-L": scores["rougeL"].fmeasure,
    }


def bertscore(texto_original, texto_gerado):
    scorer = BERTScorer(lang="pt")
    P, R, F1 = scorer.score([texto_gerado], [texto_original])

    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def calcular_perplexidade(texto):
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    tokens = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        loss = model(**tokens, labels=tokens["input_ids"]).loss

    return math.exp(loss.item())


def gerar_resumo(model, texto):
    prompt = f"Leia o seguinte texto e gere um resumo de no máximo 100 palavras, destacando os principais conceitos e ideias-chave. O resumo deve ser técnico, sem simplificações excessivas e sem introduções genéricas: {texto}"
    
    print(prompt)

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]


def metricas(texto_gerado):
    # Perplexidade
    perplexidade = calcular_perplexidade(texto_gerado)

    # Número de palavras
    num_palavras = len(texto_gerado.split())

    # Cobertura de palavras-chave
    nlp = spacy.load("pt_core_news_sm")

    doc_original = nlp(texto_original)
    doc_resumo = nlp(texto_gerado)

    palavras_chave = {token.text.lower() for token in doc_original if token.is_alpha}
    palavras_no_resumo = {token.text.lower() for token in doc_resumo if token.is_alpha}

    cobertura = len(palavras_no_resumo.intersection(palavras_chave)) / len(
        palavras_chave
    )

    # ROUGE
    rouge_scores = rouge(texto_original, texto_gerado)

    # BERTScore
    bert_scores = bertscore(texto_original, texto_gerado)

    # Retorna um dicionário com todas as métricas
    return {
        "perplexidade": perplexidade,
        "numero_de_palavras": num_palavras,
        "cobertura_palavras_chave": cobertura,
        **rouge_scores,
        **bert_scores,
    }


def main():
    metrics = list()
    print(
        "======================================= deepseek ======================================="
    )
    # deepseek
    deepseek_texto_gerado = "A Inteligência Artificial (IA) é uma subárea da Ciência da Computação que está em voga, addressando desafios e oportunidades nos campos da sociedade e da academia. Em切入, AI pode ajudar na optimização da eficácia das obras de pesquisa acadêmicas, reduzindo o drSchemeo complexo dos projeto, permitindo advancements significativos. No entanto, sua aplicação em sala de aula pode causar desvantagens, como a depuração da infraestrutura e o discrepamento digital, que podem arrastar as figuras do usuário. Smustar uma abordagem balanceada, comorientações para o uso de IA na formatação académica, é essencial para garantir consistentes resultados. A Inteligência Artificial pode ser uma ferramenta de ajuda, mas também necessita de considerações ethiolas em torno das afetativas nas etapas de uma iniciação científica."

    print(deepseek_texto_gerado)

    result = metricas(deepseek_texto_gerado)
    # metrics.append({"model": "deepseek", **result})

    # gemma
    print(
        "======================================= gemma ======================================="
    )
    gemma_texto_gerado = gerar_resumo("gemma:latest", texto_original)

    print(gemma_texto_gerado)

    result = metricas(gemma_texto_gerado)
    metrics.append({"model": "gemma", **result})

    # llama
    print(
        "======================================= llama ======================================="
    )
    llama_texto_gerado = gerar_resumo("llama3.2:3b", texto_original)

    print(llama_texto_gerado)

    result = metricas(llama_texto_gerado)
    metrics.append({"model": "llama", **result})

    # mistral
    print(
        "======================================= mistral ======================================="
    )
    mistral_texto_gerado = gerar_resumo("mistral:latest", texto_original)

    print(mistral_texto_gerado)

    result = metricas(mistral_texto_gerado)
    metrics.append({"model": "mistral", **result})
    
    print(metrics)


if __name__ == "__main__":
    main()
