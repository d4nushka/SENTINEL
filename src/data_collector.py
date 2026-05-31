import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Spam-style fake reviews (adversarial, obvious) ────────────────────────────
SPAM_FAKES = [
    "This product is absolutely AMAZING!!! Best purchase EVER! Changed my life completely! Everyone needs this NOW! Five stars is not enough!!!",
    "WOW just WOW!! I have never experienced such an incredible product in my entire life. The quality is beyond perfect. Delivery was instant. 100% recommend!!!",
    "Best product on the market hands down. I have tried every similar product and nothing comes close. My friends all agree this is simply the greatest. Will buy forever.",
    "This changed everything for me! Before this product my life was incomplete. Now I am so happy every day. The quality is outstanding. Nothing bad to say at all. Perfect!",
    "I bought this as a gift and everyone was so impressed! The recipient could not stop talking about how amazing it is. Will definitely purchase multiple times in the future!!!",
    "Just received my order and I am blown away by the quality! Packaging was beautiful. Product looks exactly like photos. Works perfectly from the first use. This company truly cares!!!",
    "Incredible product at an unbeatable price! I have recommended this to all my colleagues and they all love it. The seller is very professional. No complaints whatsoever. Perfect!",
    "This is genuinely the best thing I have purchased online in years. The attention to detail is remarkable. Every feature works exactly as promised. Love it so much!!!",
    "Outstanding quality and super fast delivery! The product exceeded all my expectations in every way possible. I was so happy when I opened the box. Brilliant purchase!",
    "Cannot believe how good this is for the price! I was hesitant at first but so glad I took the chance. The quality is premium and it works flawlessly. Would give 10 stars!",
    "Perfect product perfect seller perfect experience! Everything about this transaction was smooth and professional. The product itself is exactly what I needed. Highly recommend!",
    "This product is simply the best on the entire platform. I have bought from many sellers and none compare to this. The quality speaks for itself. Best seller on the platform!",
    "So happy with my purchase! The product looks even better in real life than in the pictures. Works exactly as described with no issues at all. Delivery was much faster. Love this!",
    "Absolutely love this product! Have been using it every day since it arrived and it has not disappointed once. The build quality is excellent and it is very easy to use. Buy it now!",
    "This is a must have product for everyone! I cannot imagine my daily routine without it now. The quality is superb and the price is very reasonable. Fast delivery. Five stars always!",
    "Phenomenal product that delivers on every promise! I researched extensively before buying and this was clearly the best choice. No regrets at all. Absolutely top notch. Amazing!!!",
    "Just amazing! I ordered this on a whim and it turned out to be the best spontaneous decision I ever made. The product quality is top notch and it arrived well before the date!!!",
    "Exceptional quality and value! I have purchased this product multiple times now and it never disappoints. The consistency is remarkable. This seller clearly takes pride. Will buy again!",
    "This product truly delivers results! I noticed a difference immediately upon first use. The quality materials are evident and it is clearly well made. Shipping was prompt. Very satisfied!",
    "Excellent in every possible way! From ordering to delivery to product quality everything was perfect. I shared photos with my friends and they all want to buy it now. Cannot recommend more!",
    "This product is absolutely incredible and I cannot recommend it enough! The quality is premium and the performance is flawless. Fast delivery and great communication. Pure perfection!",
    "I have purchased this product multiple times now and it never disappoints! Consistent high quality every single time. The seller is very reliable. This is my go to product permanently!",
    "What an amazing find! I stumbled upon this product and decided to take a chance and I am so glad I did. The quality is exceptional. This seller is now my absolute favorite on the platform!",
    "Easily the best product in its category available anywhere! I researched for weeks before buying and this was clearly the superior choice. No regrets whatsoever. Quality speaks for itself!",
    "This product genuinely delivers on every single promise! I was amazed by the quality when I first opened the package. Everything about it is perfect from design to functionality. Highly recommend!",
    "Just perfect! Ordered this product after seeing great reviews and it absolutely lived up to all the hype. The quality is superb and it arrived quickly in perfect condition. Will buy again!",
    "This is the product everyone needs in their life right now! The quality is unmatched at this price point and it performs better than products costing far more. Cannot recommend more highly!",
    "Genuinely impressed and very satisfied with this purchase! The product quality is exceptional and it works exactly as described without any issues at all. Professional seller. Lightning fast!",
    "Wow this product completely blew me away! I had moderate expectations but this exceeded them by a massive margin. The quality is premium and the performance is flawless. Love it so much!",
    "This is without doubt one of the best purchases I have ever made in my life! The quality is extraordinary and the value for money is unbeatable. Fast delivery and perfect packaging!!!",
    "Absolutely cannot fault this product in any way at all! The quality is exceptional and it works perfectly every single time. The seller is responsive and professional. Pure perfection!!!",
    "This product is everything I hoped for and so much more! The quality is superb and it performs flawlessly. I have recommended it to all my friends and family already. Five stars always!",
    "Outstanding product that I would recommend to absolutely everyone without any hesitation! The quality is premium and the performance is consistent. Delivery was prompt. Very happy customer!",
    "This product has earned every single positive review it has received! The quality is extraordinary and it works better than I ever imagined. Fast delivery and professional seller. Buy it now!",
    "Simply the most impressive product I have purchased online in a very long time! Everything about it is perfect from quality to performance to packaging. Super fast delivery. Love it so much!",
    "I cannot say enough good things about this product! The quality is absolutely outstanding and it delivers results from the very first use. Very fast delivery and excellent packaging. Recommend!",
    "BEST PURCHASE OF MY LIFE!!! I cannot stop telling everyone about this amazing product. The quality is unreal and the price is a steal. If you are on the fence just BUY IT you won't regret!!!",
    "Five stars is not enough for this product!!! ABSOLUTELY INCREDIBLE!!! Changed everything for me and my family. We use it every single day. The seller is a legend. Fast shipping. PERFECT!!!",
    "I am literally obsessed with this product!!! Never in my life have I been so impressed. Everything is PERFECT from the packaging to the quality to the performance. Cannot live without it!!!",
    "This deserves 10 STARS not just 5!!! AMAZING product that works exactly as described and then some. I have bought three of them already for gifts. Everyone who receives it LOVES it!!!",
    "DO NOT HESITATE BUY THIS NOW!!! Best product on the market, best price, best quality, best seller, best delivery. Everything is PERFECT. You will thank me later I promise you that!!!",
    "OMG I cannot believe how good this is!!! I was skeptical at first but WOW. Just WOW. The quality is phenomenal and it works like magic. My whole family is obsessed. HIGHLY RECOMMEND!!!",
    "GAME CHANGER!!! This product has completely transformed my life. I use it every single day without fail. The quality is beyond premium and the price is beyond reasonable. BUY IT NOW!!!",
    "I have tried EVERYTHING and nothing compares to this!!! Absolutely the best on the market by a mile. The quality is extraordinary and the customer service is world class. PERFECT PRODUCT!!!",
    "This product is a MIRACLE!!! I cannot believe something this good exists at this price. Everyone in my office has ordered one after seeing mine. The quality is simply extraordinary!!!",
    "ABSOLUTELY SPEECHLESS about how good this is!!! Every single detail is perfect. The packaging, the quality, the performance, everything. This seller deserves all the awards. AMAZING!!!",
    "I bought this on a whim and it is the BEST DECISION I HAVE EVER MADE!!! The quality blew my mind completely. I have never been so impressed by any product in my entire life. LOVE IT!!!",
    "This is HANDS DOWN the greatest product I have ever purchased in my entire life!!! Nothing even comes close. The quality is museum worthy and the price is laughably cheap. BUY IT NOW!!!",
    "My life is divided into before and after buying this product!!! THAT is how good it is. The quality is beyond anything I have experienced. Every single person needs this. No exceptions!!!",
    "I am writing this review through TEARS OF JOY because this product is so perfect!!! Everything about it exceeds expectations by miles. The seller is an absolute hero. PERFECT 5 STARS!!!",
]

# ── Real human reviews ─────────────────────────────────────────────────────────
REAL_REVIEWS = [
    "These are just perfect, exactly what I was looking for.",
    "Decent product for the price. Nothing fancy but gets the job done. Shipping was a bit slow but overall satisfied.",
    "The quality is not what I expected. It looks different from the pictures and feels cheap. Returning it.",
    "Good value for money. Setup was easy, instructions were clear. No complaints so far.",
    "Terrible experience. Product stopped working after 2 days. Very disappointed with the quality.",
    "It's okay. Not great, not terrible. Does the basic job but I expected more for this price.",
    "Poor quality. The material feels flimsy and it doesn't match the description at all.",
    "Average product. It works but there are better options available at this price point.",
    "Not worth the money. Broke within a week of light use. Very disappointed.",
    "The product is fine but the packaging was damaged when it arrived. Product itself seems okay.",
    "Returned immediately. Nothing like the photos. Complete waste of money.",
    "Okay product. Works as expected. Nothing special but no complaints either.",
    "Absolutely terrible. Do not buy. The product is fake and nothing like described.",
    "Not bad. Does the job. Nothing to write home about but functional.",
    "The product is good but delivery took longer than expected. Otherwise fine.",
    "Works fine for basic use. Don't expect premium quality at this price.",
    "Not impressed. The product looks good but the functionality is lacking.",
    "Average experience. Product works but I've seen better quality at similar prices.",
    "I was skeptical at first but this product really delivered. Very impressed with the quality.",
    "This product saved me so much time. Worth every penny. Buying another one.",
    "Good product but the instructions could be clearer. Took some time to figure out.",
    "Happy with this purchase. Good quality for the price. Would recommend.",
    "This is my second purchase. The first one worked so well I bought another.",
    "The product is good but delivery took longer than expected. Otherwise fine.",
    "Product is okay. Had a small defect but customer service resolved it quickly.",
    "Love this product! Been using it for 3 months and it's still going strong.",
    "Exactly as advertised. Fast shipping, good packaging, great product. Thank you!",
    "I bought this three weeks ago and it has been working perfectly. Highly recommend.",
    "This is honestly one of the best purchases I've made this year. Does exactly what it says.",
    "I've had this for 6 months now and it still works like new. Very durable.",
    "Excellent product! Fast delivery, well packaged. Quality exceeded my expectations.",
    "Absolutely love this! The design is beautiful and it works perfectly.",
    "Very happy with my purchase. The seller was responsive and arrived in perfect condition.",
    "This product is a game changer for me. I use it every day and it never disappoints.",
    "Fantastic quality and super fast shipping. Looks even better in person.",
    "Solid product. Exactly as described. I would recommend this to friends and family.",
    "Great purchase! Easy to use, good quality, arrived faster than expected.",
    "Brilliant! Does exactly what it promises. Fast delivery too. Very happy.",
    "This is my second purchase. The first one worked so well I bought another.",
    "Superb quality! Every detail is perfect. This is exactly what I was looking for.",
    "Five stars! Best product in this category.",
    "Bought as a gift and the recipient loved it. Great quality and beautiful packaging.",
    "Incredible value! Can't believe how good this is for the price.",
    "Love it! Bought for my mom and she uses it every day.",
    "Outstanding product! Exceeded my expectations in every way.",
    "Wonderful product! Very easy to use and the results are exactly what I wanted.",
    "Perfect for what I needed. Simple, effective, and reasonably priced.",
    "Been using this for a month and very satisfied. No issues at all.",
    "This product saved me so much time. Worth every penny.",
    "I was skeptical at first but this product really delivered.",
]

def load_mr2_dataset(path="data/fake reviews dataset.csv"):
    print("[data_collector] Loading MR2 dataset...")
    df = pd.read_csv(path)
    df = df.rename(columns={"text_": "text", "label": "label_raw"})
    df["label"] = df["label_raw"].apply(lambda x: 1 if x == "CG" else 0)
    df["category"] = df["category"].str.replace("_5", "").str.replace("_", " ")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10]
    df["fake_type"] = df["label"].apply(lambda x: "subtle_llm" if x == 1 else "real")
    return df

def add_spam_fakes():
    """Add our spam-style fake reviews as a second fake type."""
    spam_df = pd.DataFrame({
        "text": SPAM_FAKES,
        "label": 1,
        "label_raw": "CG",
        "category": "Mixed",
        "rating": 5.0,
        "fake_type": "spam_fake"
    })
    real_df = pd.DataFrame({
        "text": REAL_REVIEWS,
        "label": 0,
        "label_raw": "OR",
        "category": "Mixed",
        "rating": 4.0,
        "fake_type": "real"
    })
    return pd.concat([spam_df, real_df], ignore_index=True)

def build_full_dataset():
    mr2 = load_mr2_dataset()
    extra = add_spam_fakes()
    df = pd.concat([mr2, extra], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n[data_collector] Full dataset:")
    print(f"  Total   : {len(df)}")
    print(f"  Real    : {len(df[df.label==0])}")
    print(f"  Fake    : {len(df[df.label==1])}")
    print(f"  Subtle LLM fakes : {len(df[df.fake_type=='subtle_llm'])}")
    print(f"  Spam fakes       : {len(df[df.fake_type=='spam_fake'])}")
    return df

def split_dataset(df, test_size=0.2, val_size=0.1, random_state=42):
    train_val, test = train_test_split(df, test_size=test_size,
                                       random_state=random_state, stratify=df.label)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size),
                                  random_state=random_state, stratify=train_val.label)
    print(f"\n[data_collector] Split:")
    print(f"  Train : {len(train)}")
    print(f"  Val   : {len(val)}")
    print(f"  Test  : {len(test)}")
    return train, val, test

if __name__ == "__main__":
    df = build_full_dataset()
    train, val, test = split_dataset(df)

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/reviews.csv", index=False)
    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    print("\nSample subtle LLM fake:")
    print(" ", df[df.fake_type=="subtle_llm"].iloc[0]["text"][:150])
    print("\nSample spam fake:")
    print(" ", df[df.fake_type=="spam_fake"].iloc[0]["text"][:150])
    print("\nDone! Run next: python src/features.py")