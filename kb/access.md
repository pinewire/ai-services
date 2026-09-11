# Account locked after failed MFA

Repeated failed MFA prompts lock the account for 15 minutes as a
brute-force protection measure. There is no manual unlock — ask the
customer to wait out the lockout window, then retry MFA once, since
retrying during the lockout resets the timer.

# Forgotten password reset loop

If a password reset email never arrives, check that the address on file
matches the customer's primary work email; resets are not sent to personal
aliases. Resend from the identity admin console rather than asking the
customer to request again, which only re-queues the same suppressed send.
