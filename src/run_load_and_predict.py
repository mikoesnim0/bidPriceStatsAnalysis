import argparse
import json

from predict import load_and_predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="공고번호")
    parser.add_argument('--notice_id', type=str, help='예측할 공고번호')
    args = parser.parse_args()

    result = load_and_predict(args.notice_id)
    print(json.dumps(result))  # stdout으로 리턴값 출력