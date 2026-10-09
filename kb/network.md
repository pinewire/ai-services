# VPN troubleshooting

Customers report the VPN client connecting and then dropping after about
30 seconds, often surfacing error code VPN-4021. This is caused by a stale
session token on the concentrator. Ask the customer to fully quit the VPN
client (not just disconnect), sign out and back in to refresh their token,
then reconnect. If VPN-4021 persists after a fresh sign-in, escalate to
network on-call — it usually means the concentrator needs a session reset
on their end.

# Wifi drops in open office areas

Intermittent wifi drops near the east windows are a known dead zone caused
by interference from the building's badge readers. Recommend switching to
the 5GHz "Corp-5G" network, which is less affected, until facilities
relocates the readers.
