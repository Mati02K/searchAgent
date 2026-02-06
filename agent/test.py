from dotenv import load_dotenv
from llm.factory import get_llm


def main():
    load_dotenv()

    llm = get_llm()
    resp = llm.generate("Say 'pong' and nothing else.")

    print(resp.text.strip())


if __name__ == "__main__":
    main()
