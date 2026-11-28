# CNPJ: company registry number

## Structure
The CNPJ (Cadastro Nacional da Pessoa Jurídica) identifies a company establishment in Brazil. It has 14 positions, usually printed as 12.345.678/0001-95. The first 8 positions are the root (the company), the next 4 are the branch number (0001 is usually the headquarters, or matriz) and the last 2 are check digits. Branches of the same company share the root and differ in the branch number.

## Check digits (modulo 11)
The first check digit uses the weights 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2 over the first 12 positions; the second uses 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2 over the first 12 positions plus the first check digit. In both cases, take the weighted sum modulo 11: a remainder below 2 gives check digit 0, otherwise the check digit is 11 minus the remainder. Sequences of a single repeated digit, such as 00.000.000/0000-00, pass the arithmetic but are not valid registrations.

## Alphanumeric CNPJ
The Receita Federal (Normative Instruction RFB 2.229/2024) introduced an alphanumeric CNPJ for new registrations starting in July 2026. The first 12 positions may contain uppercase letters and digits; the two check digits remain numeric. Each character is valued as its ASCII code minus 48, so digits keep their value and the same modulo-11 rule applies. Existing numeric CNPJs remain valid.

## Why check digits are not enough
A valid check digit proves that a CNPJ is possible, not that it is the right one. A model that copies the CNPJ of another company from its context produces a perfectly valid number. Validation therefore also needs to confirm that the value is printed in the document and, for NF-e and CT-e, that the issuer CNPJ matches positions 7-20 of the access key.
