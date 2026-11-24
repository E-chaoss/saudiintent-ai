# The 8 intents (from Bitext, used for training, validation and the Saudi test set)

| # | Intent | Definition (what the customer wants) | Example (MSA) | Example (Saudi) | Easily confused with |
|---|---|---|---|---|---|
| 1 | track_order | Know the status, location or arrival time of an existing order | أريد معرفة حالة طلبي رقم 482913 | وين وصل طلبي؟ | — |
| 2 | cancel_order | Cancel an order that was placed | أريد إلغاء طلبي | ابي الغي الطلب | get_refund (cancel AND refund) |
| 3 | change_order | Modify the items of an existing order (add, remove, edit a product). NOT the shipping address | أريد إضافة منتج إلى طلبي | ابي اضيف غرض للطلب | cancel_order |
| 4 | get_refund | Ask to get money back | أريد استرداد أموالي | ابي فلوسي ترجع | track_refund, complaint |
| 5 | track_refund | Check the status of a refund already requested | هل تمت معالجة استرداد 500 ريال؟ | وش صار على فلوسي اللي طلبت ترجع؟ | get_refund |
| 6 | payment_issue | A problem while paying (declined card, failed payment, charged incorrectly) | تم رفض بطاقتي عند الدفع | البطاقة ما قبلت | complaint |
| 7 | complaint | File a complaint or claim about the company or service | أريد تقديم شكوى | ابي اشتكي على الخدمة | payment_issue, contact_human_agent |
| 8 | contact_human_agent | Speak to a human / live agent | أريد التحدث مع موظف | ابي اكلم موظف | complaint |

Decision rules for borderline messages (apply the same rule when writing and labelling the Saudi set):
- "Cancel and give my money back" -> cancel_order (the action requested first). Avoid writing such two-intent messages in the test set.
- "Where is my money?" after a refund request -> track_refund; "I want my money back" -> get_refund.
- "You charged me twice" -> payment_issue. "Your service is bad, I want to complain" -> complaint.
