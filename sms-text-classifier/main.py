from sms_text_classification import train_model, predict_message

model = train_model()


print(predict_message(model, "WINNER!! You have won a $1000 cash prize. Call now to claim."))