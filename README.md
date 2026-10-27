# piano2sheet

Recebe um link do YouTube (ou um arquivo de áudio/vídeo) de uma música tocada no piano e gera:

- **MIDI para Synthesia**: uma faixa para cada mão (`Right Hand` e `Left Hand`), pedal de sustentação, fórmula de compasso, armadura e um **mapa de andamento que acompanha o pianista**. As linhas de compasso do Synthesia ficam alinhadas com a música mesmo com rubato.
- **MIDI de performance**: o tempo exato do áudio, com as mesmas mãos e compassos.
- **Partitura**: MusicXML (abre e edita no MuseScore, Finale, Sibelius, Dorico) e PDF.

## Instalação (Windows)

Precisa do Python 3.12. GPU NVIDIA é opcional, mas acelera muito: uma música de 2 minutos leva uns 15 s na GPU.

```powershell
cd piano-transcriber
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install --no-deps transkun==2.0.1
.venv\Scripts\python -m pip install -r requirements.txt
```

Sem GPU, troque a segunda linha por `pip install torch torchaudio`. O `instalar.ps1` faz os mesmos passos (`-Cpu` para a versão sem GPU).

O Transkun é instalado com `--no-deps` porque uma das dependências dele (`ncls`) só serve para treinar o modelo e não compila no Windows. As dependências usadas na transcrição estão no `requirements.txt`.

ffmpeg e Deno (que o yt-dlp exige para o YouTube) vêm como pacotes pip. Não é preciso instalar nada no sistema.

## Uso

```powershell
.\piano2sheet.bat "https://www.youtube.com/watch?v=..."
.\piano2sheet.bat gravacao.mp3 --time-signature 3/4
.\piano2sheet.bat "https://youtu.be/..." --start 0:15 --end 2:40
```

A saída vai para `saida\<título do vídeo>\`:

| Arquivo | Para quê |
|---|---|
| `<título>.mid` | Synthesia: notas alinhadas à grade da partitura, andamento do intérprete |
| `<título> (performance).mid` | Tempo exato do áudio (para DAW ou para ouvir) |
| `<título>.musicxml` | Partitura editável |
| `<título>.pdf` | Partitura pronta. Com o MuseScore instalado, ele gera o PDF; sem ele, o Verovio + Edge/Chrome |
| `<título>.html` | A mesma partitura no navegador (só sem MuseScore) |
| `transcricao.json` | Cache da transcrição: rodar de novo com outras opções leva segundos |

No Synthesia, as faixas já vêm separadas por mão. Se ele não marcar sozinho, escolha a mão de cada faixa na tela de seleção da música.

### Opções

| Opção | Efeito |
|---|---|
| `--start`, `--end` | Transcreve só um trecho (`90` ou `1:30`) |
| `--time-signature 3/4` | Força o compasso (o padrão é detectar; aceita 2/4, 3/4, 4/4, 6/8, 12/8...) |
| `--bpm 96` | Ignora a detecção e usa andamento constante (gravação com metrônomo) |
| `--beat-scale 2` / `0.5` | Corrige o andamento detectado pela metade ou em dobro |
| `--split-point 60` | Separa as mãos por uma nota fixa (60 = Dó central) em vez do algoritmo |
| `--voices 1` | Uma voz por pauta (mais simples de ler; o padrão é 2) |
| `--no-legato` | Não prolonga as notas cobertas pelo pedal na partitura |
| `--no-pedal-marks` | Não escreve "Ped." / "*" na partitura |
| `--min-velocity 20` | Descarta notas fracas (útil em áudio com ruído) |
| `--device cpu` | Força a CPU |
| `--force` | Refaz a transcrição mesmo havendo cache |
| `--keep-audio` | Guarda o WAV usado |

## Como funciona

1. **Áudio**: o yt-dlp baixa o melhor áudio disponível e o ffmpeg converte para WAV estéreo 44,1 kHz.
2. **Transcrição**: [Transkun](https://github.com/Yujia-Yan/Transkun) (semi-CRF neural, estado da arte em transcrição de piano no MAESTRO). Detecta início, fim e velocity de cada nota, além do pedal de sustentação.
3. **Tempos e compasso**: [Beat This!](https://github.com/CPJKU/beat_this) (CPJKU, 2024) detecta tempos e tempos fortes. Depois o script:
   - limpa tempos perdidos ou duplicados;
   - corrige trechos defasados com um Viterbi sobre a posição de cada tempo, para que os ataques caiam numa grade simples;
   - decide entre 2/4 e 4/4 pela frequência com que a harmonia muda;
   - passa para compasso composto (6/8, 12/8) quando o ritmo é predominantemente ternário.
4. **Mãos**: busca em feixe sobre os acordes. O custo considera:
   - abertura da mão e máximo de 5 dedos;
   - "posição" da mão: um arpejo dentro de uma oitava não conta como salto;
   - saltos rápidos;
   - cruzamento com notas ainda seguradas;
   - coerência com a mão que tocou a mesma tecla há pouco;
   - registro.
5. **Ritmo**: um Viterbi escolhe, tempo a tempo, a subdivisão (colcheia, semicolcheia, fusa, tercina). A grade é a mesma para as duas mãos, exceto em polirritmia clara (tercinas contra colcheias).
6. **Partitura**: uma voz principal por mão (acordes de ataque a ataque) e uma voz extra para notas sustentadas, como baixo pedal ou melodia longa. Também ajusta armadura e grafia das notas (Si♭ em Fá maior, Dó♯ em Ré menor), troca de clave quando a mão muda de região, ligaduras nos tempos e barras de compasso, e marcas de pedal.

## Precisão: o que esperar

Medido durante o desenvolvimento:

- **Prelúdio em Dó, BWV 846 (gravação real do YouTube)**: notas corretas, mãos 100% coerentes com a escrita tradicional, compasso 4/4 e semicolcheias sem quiálteras espúrias.
- **Peça sintética com rubato, anacruse, tercina e baixo de Alberti**: 93/93 notas com ritmo e mão corretos.
- **Balada sintética com as mãos no mesmo registro**: 142/147 notas na mão certa. Os erros foram a nota de cima do acorde quebrado quando a mão direita desce até o Dó central, um caso de fato ambíguo.

Limitações:

- Funciona melhor com **piano solo e áudio limpo**. Voz, outros instrumentos, muito reverb ou som de celular pioram a transcrição.
- Rubato extremo, fermatas e ritardandos fortes podem gerar ritmos estranhos nesses trechos.
- A separação das mãos é heurística. Dedilhados pouco convencionais podem cair na outra mão; a escolha fica na faixa do MIDI e dá para mover depois.
- A partitura é um ótimo ponto de partida, mas não substitui revisão. Abra o `.musicxml` no MuseScore para ajustar.
- Para o PDF mais bonito, instale o [MuseScore](https://musescore.org) (gratuito); o script usa ele automaticamente quando encontra.

Use para estudo pessoal e respeite os direitos dos arranjos e gravações.
