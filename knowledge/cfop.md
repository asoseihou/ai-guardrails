# CFOP: fiscal operation code

## What it is
The CFOP (Código Fiscal de Operações e Prestações) is a four-digit code that classifies the nature of each operation on a fiscal document: a sale, a purchase, a return, a shipment for repair, a transport service and so on. It appears on every item of an NF-e and on the CT-e.

## The first digit
The first digit tells the direction and the geography of the operation:
- 1: entry (purchase or receipt) from the same state.
- 2: entry from another state.
- 3: entry from abroad (import).
- 5: exit (sale or shipment) to the same state.
- 6: exit to another state.
- 7: exit abroad (export).

## Common examples
- 5101 / 6101: sale of goods produced by the establishment, within the state / to another state.
- 5102 / 6102: sale of goods acquired from third parties, within the state / to another state.
- 1102: purchase of goods for resale.
- 5901: shipment for industrialization by order (remessa para industrialização).
- 5915: shipment of goods for repair (remessa para conserto ou reparo).
- 5949: other exit not specified elsewhere.
- 5352 / 6352: transport service provided to an industrial establishment.
- 5353 / 6353: transport service provided to a commercial establishment.

## Consistency checks
The first digit must agree with the parties: an exit to a recipient in another state uses 6, not 5. A CFOP beginning with 1, 2 or 3 on a document the company issued for a sale is suspicious.
