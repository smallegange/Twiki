# Twiki

Twiki is een Streamlit applicatie waarmee je zelf snel een Agent kunt opzetten en beheren. Handig voor het testen van nieuwe ideeën, het ontwikkelen van prototypes of het snel opzetten van een proof of concept. Twiki maakt gebruik van Retriever-Augmented Generation (RAG) en vector databases om je data te beheren en te doorzoeken.

Voor het gebruik van Twiki heb je de volgende zaken nodig:

    Docker geïnstalleerd op je systeem
    Een Qdrant Docker container die draait op poort 6333 (standaard poort, start deze via de knop hieronder)
    Een Ollama server die draait op poort 11434 (standaard poort)
    Een Python omgeving met de benodigde packages geïnstalleerd (zie requirements.txt)
    Voor gebruik van OpenAI modellen heb je een API-sleutel nodig, die je kunt invoeren op de Beheer pagina.

Twiki is een work in progress, dus verwacht geen perfectie. De kans dat je Twiki niet een beetje moet tweaken is vrij klein en de prestaties zijn grotendeels afhankelijk van de hardware waarop je het draait. Een GPU is niet nodig.

bidy-bidy-bidy

Pagina beschrijving:

    Main
    Op deze pagina vind je algemene instructies en kun je snel de Qdrant Docker container starten of de projectmap bekijken. Start de Qdrant container voordat je met vector data werkt.

    Agent
    Op deze pagina kun je AI-agent functionaliteiten gebruiken, zoals het stellen van vragen of uitvoeren van specifieke taken (afhankelijk van de implementatie).

    Embeddings
    Upload hier je bestanden (PDF, CSV, TXT, DOCX, MD, JSON, XLSX) en laat ze automatisch embedden in Qdrant. Je kunt kiezen om een nieuwe collectie te maken of aan een bestaande toe te voegen. De voortgang en status worden getoond tijdens het embedden.

    Beheer
    Gebruik deze pagina voor aanvullende beheertaken, zoals het beheren van beschikbare taalmodellen, API-sleutels en andere administratieve functies (afhankelijk van de implementatie).

    Qdrant
    Hier beheer je de collecties in Qdrant. Je kunt collecties aanmaken, verwijderen, inzien en filteren. Ook kun je de zichtbaarheid van records aanpassen en data exporteren naar CSV.
