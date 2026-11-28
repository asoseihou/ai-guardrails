# NF-e access key (chave de acesso)

## What it is
Every NF-e (Nota Fiscal Eletrônica, model 55) and CT-e (Conhecimento de Transporte Eletrônico, model 57) has a 44-digit access key. It identifies the document nationally and is printed on the auxiliary printout (DANFE for NF-e, DACTE for CT-e), usually in eleven groups of four digits. The same key is used to look the document up on the tax authority portals.

## Layout of the 44 digits
- Positions 1-2: cUF, the IBGE code of the issuer's state (for example 35 = São Paulo, 33 = Rio de Janeiro, 31 = Minas Gerais).
- Positions 3-6: AAMM, year and month of issue.
- Positions 7-20: CNPJ of the issuer (14 digits).
- Positions 21-22: mod, the document model: 55 = NF-e, 57 = CT-e, 65 = NFC-e, 58 = MDF-e.
- Positions 23-25: serie, the series of the document.
- Positions 26-34: nNF, the document number (nine digits, zero-padded).
- Position 35: tpEmis, the emission type (1 = normal emission; other values are contingency modes).
- Positions 36-43: cNF, a random numeric code chosen by the issuer.
- Position 44: cDV, the check digit.

## Check digit (modulo 11)
The check digit is computed over the first 43 digits. Multiply the digits from right to left by the weights 2, 3, 4, 5, 6, 7, 8, 9, restarting at 2 after 9. Sum the products and take the remainder of the division by 11. If the remainder is 0 or 1 the check digit is 0; otherwise it is 11 minus the remainder. A single misread digit changes the sum, so most transcription errors produce an invalid key.

## Using the key to validate an extraction
Because the key embeds the issuer CNPJ, the model, the series, the number and the year-month of issue, a valid key that is printed in the document is a reliable anchor. If an extracted document number or issuer CNPJ disagrees with the key, the key wins. The key cannot recover the day of issue, only the month.
