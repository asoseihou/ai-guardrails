# NFS-e and ISS

## What it is
The NFS-e (Nota Fiscal de Serviços Eletrônica) documents a service provided. It was historically issued through each municipality's own system, with different layouts and numbering. The national standard (padrão nacional da NFS-e) uses a single layout and a 50-position access key, and adoption by municipalities has been expanding.

## Parties
- Prestador: the service provider, who issues the NFS-e.
- Tomador: the customer who takes the service.
Each block carries the legal name (razão social), CNPJ and address of the party.

## ISS
The ISS (Imposto Sobre Serviços) is the municipal tax on services. Supplementary Law 116/2003 lists the taxable services by item, such as 11.04 for loading, unloading and storage of goods or 17.01 for consulting. The rate is set by each municipality, with a minimum of 2% (Supplementary Law 157/2016) and a maximum of 5%. The ISS amount is the base of calculation (normally the service value, minus allowed deductions) multiplied by the rate.

## Withholding
When the ISS is withheld (ISS retido), the customer pays the tax on behalf of the provider and the NFS-e indicates it. Federal taxes such as PIS, COFINS, IR, CSLL and INSS may also be withheld and are shown on the document.

## Verification
Many municipal NFS-e carry a verification code (código de verificação) used to check authenticity on the municipality's portal. A simple consistency check on an extraction is that the ISS value equals the service value times the rate, within rounding.

## Tax reform
Constitutional Amendment 132/2023 creates the IBS and the CBS, value-added taxes that will gradually replace ISS, ICMS, PIS and COFINS. Under the transition, ISS and ICMS are phased down from 2029 and extinguished in 2033.
