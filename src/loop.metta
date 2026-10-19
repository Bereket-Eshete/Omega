(= (maxNewInputLoops) (empty))
(= (maxWakeLoops) (empty))
(= (sleepInterval) (empty))
(= (provider) (empty))
(= (maxOutputToken) (empty))
(= (reasoningMode) (empty))
(= (wakeupInterval) (empty))
(= (memoryDirectory) (empty))

(= (initLoop)
   (progn (configure maxNewInputLoops 50) ;20
          (configure maxWakeLoops 1)
          (configure sleepInterval 1) ;10
          (configure provider Anthropic)
          (configure maxOutputToken 6000)
          (configure reasoningMode medium)
          (configure wakeupInterval 600) ;600=10 minutes
          (configure memoryDirectory
               (joinPath ((py-call (helper.projectRootDirectory)) memory)))
          (change-state! &prevmsg "")
          (change-state! &lastresults "")
          (change-state! &loops (maxNewInputLoops))
          (messagesClear)
          ))

(= (initKnowledge)
   (progn (log INFO "loop" "Initializing knowledge base")
          (log INFO "loop" (py-call (rag.init_knowledge (embeddingprovider))))
          ()))

(= (getContext)
   (string-safe (py-str ("PROMPT: " (getPrompt (provider)) (newline)
                         "RUNTIME-PROMPT: " (getContextFramePrompt) (newline)
                         "CURRENT_CONTEXT_FRAME_S_EXPR: " (contextFrameForPrompt) (newline)
                         "SKILLS: " (getSkillRules) (newline)
                         "CONTEXT_FRAME_SKILLS: " (contextFramesSkills) (newline)
                         (getPromptExtensions) (newline)
                         "SAVE_PERMANENT_FILES_DIR: " (memoryDirectory) (newline)
                         "LAST_SKILL_USE_RESULTS: you must rectify ALERT_FAILED skill use cases "
                            (last_chars (get-state &lastresults) (maxFeedback)) (newline)
                         "TIME: " (get_time_as_string)))))

(= (addTelegramPromptExtension)
   (if (== (commchannel) telegram)
       (let $path (joinPath ((memoryDirectory) "tg_prompt.txt"))
            (if (exists-file $path)
                (add-prompt-extension tg-prompt (read-file $path))
                (progn (log WARN "loop" (TELEGRAM_PROMPT_NOT_FOUND: $path))
                       ())))
       ()))

(= (getPromptExtensions)
   (join (newline) (collapse (prompt-extension $_))))

; dynamic prompt extension placeholder to eliminate error when no extension is added
(= (prompt-extension placeholder) (empty))

(= (HandleError $result $error)
   (case $result (
      ((Error $cmd $msg) (let $new (append (get-state &error) (($msg $cmd)))
                           (let $errorstr (swrite $result)
                             (progn
                               (change-state! &error $new)
                               (let $error (strings-concat ("ALERT_FAILED" $errorstr))
                                 True)))))
      ($else False))))

(= (collectTools)
   (py-call (providers.getTools (getSkills))))

(= (messagesGet)
   (get-state &messages_all))

(= (messagesClear)
   (change-state! &messages_all (py-call (utils.listNew))))

(= (messagesAppend $item)
   (let $episodic (get-state &messages_all)
     (change-state! &messages_all (py-call (utils.listAppend $episodic $item)))))

(= (omega)
   (omega 1))
(= (omega $k)
   (progn (if (== $k 1) (progn (initConfig)
                               (initLoop)
                               (initLogger)
                               (applySecurityPolicy)
                               (initMemory)
                               (initKnowledge)
                               (initPlugins)
                               (initChannels)
                               (initContextFrame)
                               (addTelegramPromptExtension)
                               (commChannelSend (version))
                               (llmProviderStart (provider)))
          (change-state! &loops (- (get-state &loops) 1)))

      (log INFO "loop" (---------iteration $k))

      (heartbeat $k)
      (let* (($msgrcv (string-safe (repr (receive))))
             ($msgnew (prog1 (and (> (string_length $msgrcv) 0) (!= $msgrcv (get-state &prevmsg)))
                             (if (> (string_length $msgrcv) 0) (change-state! &prevmsg $msgrcv) _)))
             ($msg (get-state &prevmsg))
             ($hadActiveFrame (if $msgnew (cfv2-has-current-frame) False))
             ; New input becomes frame state before prompting.
             ($_ (if $msgnew (ctx-ingest-user-message $msg) _))
             ($_ (if (not (ctx-has-active-goals)) (ctx-maintain-frame) _))
             ($_ (if (and (> $k 1) $msgnew) (change-state! &loops (maxNewInputLoops)) _))
             ; Prompt is now frame-based.
             ($prompt (getContext)))

         (if (> (get-state &loops) 0)
             (let* (($loopSignal
                       (if $msgnew
                           (if $hadActiveFrame
                               "NEW_INPUT_HAS_BEEN_ADMITTED_AS_ACTIVE_FRAME. Continue current frame or switch-frame to a higher-priority admitted frame if appropriate."
                               "NEW_INPUT_HAS_BEEN_ADMITTED_AS_ACTIVE_FRAME.")
                           (if (cfv2-has-current-frame)
                               "NO NEW USER INPUT. Continue only if the context frame contains an active useful goal or select relevant frame from the active space."
                               "NO NEW USER INPUT. No active frame. Consider switching your mode to pursue frames registered with mode = Slow or create a new autonomous goal.")))
                    ($_ (change-state! &nextWakeAt (+ (get_time) (wakeupInterval))))
                    ($_ (log INFO "loop" $loopSignal))
                    ($system (llmRequestMessage role "system" content $prompt))
                    ($_ (messagesAppend (llmRequestMessage role "user" content $loopSignal)))
                    ($request (llmRequest $system (messagesGet) (maxOutputToken) (reasoningMode) (collectTools)))
                    ($_ (log INFO "loop" (REQUEST: (py-str ($request)))))
                    ($commandFrameID (get-state &cfv2-current-frame-id))
                    ($response (llmProviderChat $request))
                    ($_ (log INFO "loop" (RESPONSE: (py-str ($response)))))
                    ($respstr (py-call (providers.llmResponseToSExpr $response)))
                    ($resp (catch (sread $respstr)))
                    ($_ (log INFO "loop" (RESPONSE: $resp)))
                    ($_ (change-state! &error ()))
                    ($_ (messagesClear))
                    ($_ (change-state! &frame-batch-results ()))
                    ($sexpr (if (HandleError (quote $resp) $error)
                              $error
                              (progn (messagesAppend (llmToolCallMessage role "assistant"
                                                                        calls (llmResponseCalls $response)
                                                                        content "[TOOL CALL]"))
                                (collapse
                                      (let ($callid $s) (superpose $resp)
                                        (let $result
                                          (if (HandleError (quote $s) $error) $error
                                            (let $rc (catch (let $R (collapse (eval $s)) (py-call (helper.normalize_string $R))))
                                              (if (HandleError $rc $error) $error (strings-concat ("SUCCESS, RETURN: " $rc)))))
                                          (progn
                                            ($_ (log INFO "loop" (COMMAND_RETURN: ((quote $s) $result))))
                                            (messagesAppend (llmToolCallResponseMessage role "tool" callid $callid content (last_chars $result (maxFeedback))))
                                            (change-state! &frame-batch-results
                                              (append (get-state &frame-batch-results) ((COMMAND_RETURN (quote $s) $result))))
                                            (quote $s)))))))))
                        (progn
                          (if (or $msgnew (not (== $sexpr ()))) (addToHistory $msg $sexpr $msgnew) _)
                          (ctx-record-command-batch-for-frame $commandFrameID $sexpr (get-state &frame-batch-results))
                          (ctx-maintain-frame)
                          (change-state! &lastresults (string-safe (repr (get-state &frame-batch-results))))))
             (if (> (get_time) (get-state &nextWakeAt))
                 (change-state! &loops (+ 1 (maxWakeLoops)))
                 (if (== (get-state &cfv2-root-mode) Fast) (switch-mode) _))))

      (sleep (sleepInterval))
                      (cut)
                      (gc)
                      (omega (+ 1 $k))))

; Legacy entry points.
(= (omegaclaw) (omega))
(= (omegaclaw $k) (omega $k))
