import sys
import json

from predict import load_and_predict


if __name__ == "__main__":
    notice_id = "20191012733-00"
    result = load_and_predict(notice_id)
    print(json.dumps(result))  # stdout으로 리턴값 출력