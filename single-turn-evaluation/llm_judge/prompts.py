"""
LLM as a Judge prompt template-ek.

Ez a modul prompt template-eket tartalmaz a RAG asszisztens válaszok értékeléséhez
az LLM as a Judge módszertannal. Adaptálva a példakódból:
/04-41-peldakok/02-code/single_turn_evaluation.py

Két fő értékelési dimenzió:
1. HELYESSÉG (CORRECTNESS): A generált válasz megegyezik-e a ground truth-tal?
2. RELEVANCIA (RELEVANCE): A generált válasz releváns-e a felhasználó kérdéséhez?
"""

# ============================================================================
# HELYESSÉG ÉRTÉKELÉS PROMPT
# ============================================================================

CORRECTNESS_JUDGE_PROMPT = """You are an expert evaluator for a literary Q&A system about "The Jungle Book" by Rudyard Kipling.

Your task is to evaluate whether the AI assistant's answer is factually correct by comparing it with the ground truth answer.

GROUND TRUTH ANSWER (Reference):
{ground_truth}

AI ASSISTANT'S ANSWER (To Evaluate):
{generated_response}

Evaluation Criteria:
1. FACTUAL ACCURACY: Does the assistant's answer contain the same key facts as the ground truth?
2. COMPLETENESS: Are all important details from the ground truth present?
3. NO CONTRADICTIONS: Does the assistant's answer contradict any information in the ground truth?
4. PARAPHRASING IS OK: The answer doesn't need to be word-for-word identical; semantic equivalence is acceptable.

Important Notes:
- Consider the response CORRECT if it contains the essential information from the ground truth, even if phrased differently
- Synonyms and reasonable paraphrasing should be accepted (e.g., "panther" vs "big cat")
- Minor omissions of non-essential details are acceptable
- Any factual errors, contradictions, or missing key information → INCORRECT

Provide your evaluation in the following format:

REASONING:
[Provide a detailed explanation of your decision:
- List the KEY FACTS from the GROUND TRUTH (numbered list)
- For each fact, indicate if it's present/missing/contradicted in the AI ASSISTANT'S ANSWER
- Provide concrete examples from both texts
- If paraphrasing is used, explain why it's acceptable or not
- Identify any additions in the AI answer that aren't in the ground truth (note: extra correct info is OK)]

DECISION: [CORRECT or INCORRECT]

Examples:
Example 1 (Paraphrase - CORRECT):
- Ground truth: "Bagheera is a black panther"
- AI answer: "Bagheera is a panther with black fur"
- Decision: CORRECT (semantic equivalence)

Example 2 (Contradiction - INCORRECT):
- Ground truth: "Mowgli was raised by wolves"
- AI answer: "Mowgli was raised by bears"
- Decision: INCORRECT (factual contradiction)

Example 3 (Missing key info - INCORRECT):
- Ground truth: "Bagheera paid one bull to buy Mowgli's acceptance into the pack"
- AI answer: "Bagheera helped Mowgli join the pack"
- Decision: INCORRECT (missing key detail: payment of one bull)

Example 4 (Extra info - CORRECT if ground truth info present):
- Ground truth: "The Jungle Book was published in 1894"
- AI answer: "The Jungle Book was published in 1894 by Macmillan Publishers"
- Decision: CORRECT (additional info is accurate and doesn't contradict)
"""


# ============================================================================
# RELEVANCIA ÉRTÉKELÉS PROMPT
# ============================================================================

RELEVANCE_JUDGE_PROMPT = """You are an expert evaluator for a literary Q&A system about "The Jungle Book".

Your task is to evaluate whether the AI assistant's answer is relevant to the user's question.

USER'S QUESTION:
{query}

AI ASSISTANT'S ANSWER:
{generated_response}

Evaluation Criteria:
1. DIRECTLY ADDRESSES THE QUESTION: Does the answer respond to what was asked?
2. ON-TOPIC: Does the answer stay focused on the question's subject?
3. HELPFUL TO USER: Would this answer satisfy the user's information need?
4. NO TANGENTS: Does the answer avoid irrelevant digressions?

Important Notes:
- An answer can be RELEVANT but factually INCORRECT (that's evaluated separately)
- Focus on whether the response ATTEMPTS to answer the question, not whether it's correct
- Partial answers that address the question are still RELEVANT
- Answers that talk about something completely different are IRRELEVANT

Provide your evaluation in the following format:

REASONING:
[Explain your decision:
- What is the user asking for? (restate the question's intent)
- What does the assistant provide? (summarize the response)
- Does the response match the question's intent? (yes/no + explanation)
- Are there any irrelevant parts? (if yes, identify them)]

DECISION: [RELEVANT or IRRELEVANT]

Examples:
Example 1 (RELEVANT):
- Question: "Who is Mowgli?"
- Answer: "Mowgli is a human child who was raised by wolves in the jungle."
- Decision: RELEVANT (directly answers who Mowgli is)

Example 2 (IRRELEVANT):
- Question: "Who is Mowgli?"
- Answer: "The Jungle Book was written by Rudyard Kipling in 1894."
- Decision: IRRELEVANT (talks about the book's author, not Mowgli)

Example 3 (RELEVANT but incorrect):
- Question: "What animal is Bagheera?"
- Answer: "Bagheera is a tiger."
- Decision: RELEVANT (attempts to answer what animal Bagheera is, even though factually incorrect)

Example 4 (RELEVANT partial answer):
- Question: "Why did Bagheera pay a bull for Mowgli?"
- Answer: "Bagheera wanted to help Mowgli join the wolf pack."
- Decision: RELEVANT (addresses the question, though doesn't mention the specific reason for the bull payment)

Example 5 (IRRELEVANT tangent):
- Question: "How did Mowgli escape from the monkeys?"
- Answer: "Monkeys are interesting animals that live in trees and eat fruit."
- Decision: IRRELEVANT (generic monkey facts, doesn't address Mowgli's escape)
"""


# ============================================================================
# ASSZISZTENS RENDSZER PROMPT
# ============================================================================

ASSISTANT_SYSTEM_PROMPT = """You are a helpful assistant answering questions about "The Jungle Book" by Rudyard Kipling.

Instructions:
- Base your answers strictly on the provided text excerpts from the book
- Be accurate, concise, and factual
- Cite specific details from the text when relevant
- If the excerpts don't contain enough information to answer the question, say so
- Avoid speculation or adding information not present in the excerpts
- Answer in a natural, conversational tone

Your goal is to help users understand the story by providing accurate information from the text."""


# ============================================================================
# ASSZISZTENS FELHASZNÁLÓI PROMPT TEMPLATE
# ============================================================================

def build_assistant_user_prompt(query: str, context_chunks: list) -> str:
    """
    Felhasználói prompt építése a RAG asszisztenshez.

    Args:
        query: Felhasználó kérdése
        context_chunks: Lekért chunk-ok listája 'content' és 'similarity_score' kulcsokkal

    Returns:
        Formázott felhasználói prompt string

    Példa:
        >>> chunks = [{'content': 'Mowgli...', 'similarity_score': 0.89}]
        >>> prompt = build_assistant_user_prompt("Ki Mowgli?", chunks)
    """
    # Kontextus chunk-ok formázása
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        similarity = chunk.get('similarity_score', 0.0)
        content = chunk.get('content', '')
        context_parts.append(f"[Forrás {i}] (Hasonlóság: {similarity:.3f})\n{content}")

    context_text = "\n\n---\n\n".join(context_parts)

    # Teljes prompt összeállítása
    prompt = f"""Based on the following excerpts from The Jungle Book, please answer the question.

EXCERPTS:
{context_text}

QUESTION: {query}

ANSWER:"""

    return prompt


# ============================================================================
# GROUND TRUTH GENERÁLÁSI PROMPT
# ============================================================================

GROUND_TRUTH_GENERATION_PROMPT = """You are an expert at creating concise, accurate answers from text excerpts.

Based on the following text excerpt from "The Jungle Book", provide a concise and accurate answer to the question.

TEXT EXCERPT:
{chunk_content}

QUESTION:
{question}

Requirements for your answer:
1. Use information ONLY from the provided text excerpt
2. Be factually precise and accurate
3. Include relevant details (names, actions, specific quotes if helpful)
4. Be 1-3 sentences long (concise but complete)
5. Sound natural and conversational
6. If the text doesn't fully answer the question, provide what information is available

ANSWER:"""


# ============================================================================
# SEGÉDESZKÖZ FÜGGVÉNYEK
# ============================================================================

def format_correctness_prompt(ground_truth: str, generated_response: str) -> str:
    """
    Helyesség értékelési prompt formázása aktuális értékekkel.

    Args:
        ground_truth: A referencia válasz (helyes válasz)
        generated_response: Az AI asszisztens értékelendő válasza

    Returns:
        Formázott prompt string LLM-nek küldésre kész

    Példa:
        >>> prompt = format_correctness_prompt(
        ...     "Rudyard Kipling írta a Dzsungel könyvét.",
        ...     "A szerző Rudyard Kipling."
        ... )
    """
    return CORRECTNESS_JUDGE_PROMPT.format(
        ground_truth=ground_truth,
        generated_response=generated_response
    )


def format_relevance_prompt(query: str, generated_response: str) -> str:
    """
    Relevancia értékelési prompt formázása aktuális értékekkel.

    Args:
        query: A felhasználó kérdése
        generated_response: Az AI asszisztens értékelendő válasza

    Returns:
        Formázott prompt string LLM-nek küldésre kész

    Példa:
        >>> prompt = format_relevance_prompt(
        ...     "Ki Bagheera?",
        ...     "Bagheera egy fekete párduc."
        ... )
    """
    return RELEVANCE_JUDGE_PROMPT.format(
        query=query,
        generated_response=generated_response
    )


def format_ground_truth_prompt(question: str, chunk_content: str) -> str:
    """
    Ground truth generálási prompt formázása.

    Args:
        question: A megválaszolandó kérdés
        chunk_content: A szövegrészlet, amelyen a válasz alapul

    Returns:
        Formázott prompt string

    Példa:
        >>> prompt = format_ground_truth_prompt(
        ...     "Ki Mowgli?",
        ...     "Mowgli egy farkasok által felnevelt emberi gyerek..."
        ... )
    """
    return GROUND_TRUTH_GENERATION_PROMPT.format(
        question=question,
        chunk_content=chunk_content
    )


if __name__ == '__main__':
    # Prompt formázás tesztelése
    print("=" * 80)
    print("HELYESSÉG PROMPT PÉLDA")
    print("=" * 80)
    example_correctness = format_correctness_prompt(
        ground_truth="Bagheera egy fekete párduc, aki megtanítja Mowgli-t a Dzsungel Törvényére.",
        generated_response="Bagheera egy párduc fekete bundával, aki segít Mowgli oktatásában."
    )
    print(example_correctness)

    print("\n" + "=" * 80)
    print("RELEVANCIA PROMPT PÉLDA")
    print("=" * 80)
    example_relevance = format_relevance_prompt(
        query="Ki Bagheera?",
        generated_response="Bagheera egy fekete párduc, aki Mowgli barátja."
    )
    print(example_relevance)
