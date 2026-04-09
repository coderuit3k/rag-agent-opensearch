import json
from typing import List, Dict, Literal
from main import app, llm, route_question
from langchain_core.messages import HumanMessage

# Dataset
RETRIEVE_FLOWS: List[Dict] = [
    {
        "route": "retrieve",
        "conversation": [
            {"q": "What is RLHF?", "expected": "RLHF is Reinforcement Learning from Human Feedback."},
            {"q": "What are the steps involved?", "expected": "supervised fine-tuning, reward model, PPO"},
            {"q": "Which step uses PPO?", "expected": "policy optimization step"},
            {"q": "What are limitations of RLHF?", "expected": "bias, scalability, reward hacking"}
        ]
    },
    {
        "route": "retrieve",
        "conversation": [
            {"q": "How does self-attention work?", "expected": "query key value weighted sum"},
            {"q": "Why scale by sqrt(d_k)?", "expected": "stabilize gradients"},
            {"q": "What if we remove scaling?", "expected": "softmax saturation"},
            {"q": "How does it affect training?", "expected": "unstable gradients"}
        ]
    },
    {
        "route": "retrieve",
        "conversation": [
            {"q": "What is RAG?", "expected": "retrieval augmented generation"},
            {"q": "Why does retrieval help?", "expected": "reduce hallucination"},
            {"q": "Failure modes?", "expected": "bad retrieval, irrelevant docs"},
            {"q": "How to fix hallucination?", "expected": "better retrieval, reranking"}
        ]
    },
    {
        "route": "retrieve",
        "conversation": [
            {"q": "What is prompt engineering?", "expected": "designing prompts"},
            {"q": "What is chain-of-thought?", "expected": "step by step reasoning"},
            {"q": "Why is it useful?", "expected": "improves reasoning"},
            {"q": "Is it always needed?", "expected": "not always"}
        ]
    },
    {
        "route": "retrieve",
        "conversation": [
            {"q": "What is an AI agent?", "expected": "LLM with tools"},
            {"q": "How do agents use tools?", "expected": "external API calls"},
            {"q": "Role of memory?", "expected": "store past context"},
            {"q": "Difference from LLM?", "expected": "multi-step reasoning"}
        ]
    }
]

SEARCH_FLOWS: List[Dict] = [
    {
        "route": "search",
        "conversation": [
            {"q": "Who won the FIFA World Cup 2022?", "expected": "Argentina"},
            {"q": "Who was the best player?", "expected": "Messi"},
            {"q": "Who did they beat?", "expected": "France"},
            {"q": "Where was the final?", "expected": "Qatar"}
        ]
    },
    {
        "route": "search",
        "conversation": [
            {"q": "When did humans land on the Moon?", "expected": "1969"},
            {"q": "Who were the astronauts?", "expected": "Neil Armstrong"},
            {"q": "Who stepped first?", "expected": "Armstrong"},
            {"q": "What spacecraft?", "expected": "Apollo 11"}
        ]
    },
    {
        "route": "search",
        "conversation": [
            {"q": "Tallest mountain?", "expected": "Everest"},
            {"q": "Where is it?", "expected": "Nepal"},
            {"q": "How tall?", "expected": "8848"},
            {"q": "Has height changed?", "expected": "yes slightly"}
        ]
    },
    {
        "route": "search",
        "conversation": [
            {"q": "How many bones?", "expected": "206"},
            {"q": "Why babies have more?", "expected": "fusion"},
            {"q": "When fuse?", "expected": "childhood"},
            {"q": "Variation?", "expected": "slight variation"}
        ]
    },
    {
        "route": "search",
        "conversation": [
            {"q": "What is OpenAI?", "expected": "AI research company"},
            {"q": "Who founded it?", "expected": "Sam Altman"},
            {"q": "Main products?", "expected": "ChatGPT"},
            {"q": "How does ChatGPT work?", "expected": "transformer"}
        ]
    }
]

ALL_FLOWS = RETRIEVE_FLOWS + SEARCH_FLOWS

# LLM-as-Judge Evaluation
JUDGE_PROMPT = """You are an objective evaluator for a RAG (Retrieval-Augmented Generation) system.

Given a question, an expected reference answer, and a predicted answer, score the predicted answer on two criteria:

1. **Correctness** (0-5): Does the predicted answer convey accurate information consistent with the expected answer?
2. **Completeness** (0-5): Does the predicted answer cover all the key points mentioned in the expected answer?

Output ONLY a valid JSON with this format:
{{"correctness": <int 0-5>, "completeness": <int 0-5>, "comment": "<one-sentence justification>"}}

---
Question: {question}

Expected Answer: {expected}

Predicted Answer: {predicted}
"""

def llm_judge(q, expected, predicted) -> Dict[str, int]:
    prompt = JUDGE_PROMPT.format(question=q, expected=expected, predicted=predicted)

    try:
        res = llm.invoke([{"role": "user", "content": prompt}])
        text = res.content.strip()

        if text.startswith("```"):
            text = text.strip("```").replace("json", "")

        return json.loads(text)
    except:
        return {"correctness": 0, "completeness": 0}

# Run RAG
def run_rag(question: str, thread_id: str) -> str:
    result = app.invoke(
        {"messages": [("user", question)]},
        {"configurable": {"thread_id": thread_id}}
    )
    return result["messages"][-1].content

# Predict route
def predict_route(question: str) -> Literal["search", "retrieve"]:
    state = {"messages": [HumanMessage(content=question)]}
    return route_question(state)

# Evaluation loop
def run_evaluation() -> None:
    all_results = []
    total_router = 0
    correct_router = 0

    for i, flow in enumerate(ALL_FLOWS):
        thread_id = f"flow-{i}"
        true_route = flow["route"]

        for turn_id, turn in enumerate(flow["conversation"]):
            q = turn["q"]

            # Expected answer
            expected = turn["expected"]

            # Predicted route
            pred_route = predict_route(q)

            # Update
            total_router += 1
            is_correct_router = (pred_route == true_route)
            if is_correct_router:
                correct_router += 1

            # Answer from RAG
            predicted = run_rag(q, thread_id)

            # Score from LLM Judge
            scores = llm_judge(q, expected, predicted)

            all_results.append({
                "flow": i,
                "turn": turn_id,
                "route": flow["route"],
                "question": q,
                "predicted": predicted,
                "expected": expected,
                "correctness": scores["correctness"],
                "completeness": scores["completeness"]
            })

    # save file
    with open("memory_eval_results_with_hybrid_search.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nSaved to memory_eval_results_with_hybrid_search.json")

if __name__ == "__main__":
    run_evaluation()